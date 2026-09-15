// Batched deterministic MaleCNS timestep engine.
//
// One threadgroup owns a complete batch.  That makes every dependency a cheap
// device-scope threadgroup barrier instead of a command-encoder boundary.  The
// CPU ordering contract is retained where it is observable: active-list order,
// delayed-source rank, CSR edge order for awakening, and ordered accumulation
// of convergent sources.  Independent neurons and the unique targets inside a
// single source row are processed in parallel.
#include <metal_stdlib>
using namespace metal;

struct V4Control {
  long clock;
  uint n, slots, delay, rfc;
  float dt;
  ulong total_sources, total_edges, total_active_visits;
  uint max_sources, max_edges, max_active;
};

inline void v4_evolve(int i, long now, float current,
                      device float *v, device float *g,
                      device short *refractory, device long *last,
                      device const float *av, device const float *ag, float dt) {
  long d=now-last[i]; if(d<=0) return;
  int frozen=refractory[i]>0?refractory[i]-1:0;
  int skip=int(min(d,long(frozen)));
  refractory[i]=d>=refractory[i]?0:short(refractory[i]-d); d-=skip;
  if(d>0) {
    float a=d<1024?av[d]:exp(-dt*float(d)/20.0f);
    float b=d<1024?ag[d]:exp(-dt*float(d)/5.0f);
    v[i]=-52.0f+(v[i]+52.0f)*a+current*(1.0f-a)+g[i]*(a-b)/3.0f;
    g[i]*=b;
  }
  last[i]=now;
}

kernel void v4_step_batch(
    device const long *ptr [[buffer(0)]], device const int *post [[buffer(1)]],
    device const float *weight [[buffer(2)]], device float *v [[buffer(3)]],
    device float *g [[buffer(4)]], device short *refractory [[buffer(5)]],
    device const float *drive [[buffer(6)]], device float *previous [[buffer(7)]],
    device int *events [[buffer(8)]], device int *event_counts [[buffer(9)]],
    device int *spike_counts [[buffer(10)]], device int *active [[buffer(11)]],
    device uchar *active_flags [[buffer(12)]], device int *nactive_ptr [[buffer(13)]],
    device long *last [[buffer(14)]], device const float *av [[buffer(15)]],
    device const float *ag [[buffer(16)]], device uchar *changed [[buffer(17)]],
    device V4Control *control [[buffer(18)]], constant uint &steps [[buffer(19)]],
    uint lane [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]]) {
  const uint n=control->n;

  // Native neural_advance settles a changed drive once, before the batch.
  for(uint i=lane;i<n;i+=width) {
    if(drive[i]!=previous[i]) {
      v4_evolve(int(i),control->clock-1,previous[i],v,g,refractory,last,av,ag,control->dt);
      previous[i]=drive[i]; changed[i]=1;
    } else changed[i]=0;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if(lane==0) for(uint i=0;i<n;i++) if(changed[i] && !active_flags[i]) {
    active_flags[i]=1; active[nactive_ptr[0]++]=int(i);
  }
  threadgroup_barrier(mem_flags::mem_device);

  threadgroup long row_start, row_end;
  threadgroup int source_count;
  for(uint tick=0;tick<steps;tick++) {
    const long clock=control->clock;
    const uint slot=uint(clock%long(control->slots));
    const uint future=uint((clock+long(control->delay))%long(control->slots));

    // This scan is deliberately scalar: its stable order defines spike and
    // future event order.  It touches only the dynamic active set.
    if(lane==0) {
      int kept=0, original=nactive_ptr[0];
      control->total_active_visits+=ulong(original);
      control->max_active=max(control->max_active,uint(original));
      for(int k=0;k<original;k++) {
        const int i=active[k]; v4_evolve(i,clock,drive[i],v,g,refractory,last,av,ag,control->dt);
        if(refractory[i]==0 && v[i]>-45.0f) { events[future*n+uint(event_counts[future]++)]=i; spike_counts[i]++; }
        const bool can_fire=v[i]>-45.0f || drive[i]>7.0f || drive[i]+g[i]>7.0f;
        if(can_fire) active[kept++]=i; else active_flags[i]=0;
      }
      nactive_ptr[0]=kept;
      source_count=event_counts[slot];
      ulong edge_count=0;
      for(int rank=0;rank<source_count;rank++) {
        int s=events[slot*n+uint(rank)]; edge_count+=ulong(ptr[s+1]-ptr[s]);
      }
      control->total_sources+=ulong(source_count); control->total_edges+=edge_count;
      control->max_sources=max(control->max_sources,uint(source_count));
      control->max_edges=max(control->max_edges,uint(min(edge_count,ulong(0xffffffffu))));
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Convergent source rows remain rank-serial, preserving float addition
    // order.  Targets within a row are unique in the frozen MaleCNS graph.
    for(int rank=0;rank<source_count;rank++) {
      if(lane==0) {
        int source=events[slot*n+uint(rank)]; row_start=ptr[source]; row_end=ptr[source+1];
      }
      threadgroup_barrier(mem_flags::mem_device);
      for(long edge=row_start+long(lane);edge<row_end;edge+=long(width)) {
        int target=post[edge];
        v4_evolve(target,clock,drive[target],v,g,refractory,last,av,ag,control->dt);
        if(refractory[target]==0) g[target]+=weight[edge];
      }
      threadgroup_barrier(mem_flags::mem_device);
      // Re-scan the original row to reconstruct exact CSR awakening order.
      // No per-edge flag buffer is needed because targets in a row are unique
      // and acceptance leaves refractory==0.
      if(lane==0) for(long edge=row_start;edge<row_end;edge++) {
        int target=post[edge];
        if(refractory[target]==0 && !active_flags[target]) {
          active_flags[target]=1; active[nactive_ptr[0]++]=target;
        }
      }
      threadgroup_barrier(mem_flags::mem_device);
    }

    if(lane==0) {
      event_counts[slot]=0;
      int queued=event_counts[future];
      for(int q=0;q<queued;q++) {
        int i=events[future*n+uint(q)]; v[i]=-52.0f; g[i]=0.0f; refractory[i]=short(control->rfc);
      }
      control->clock=clock+1;
    }
    threadgroup_barrier(mem_flags::mem_device);
  }

  // Match native multi-step semantics: materialize once at the observation
  // boundary, not after every internal 0.1-ms tick.
  const long boundary=control->clock-1;
  for(uint i=lane;i<n;i+=width) v4_evolve(int(i),boundary,drive[i],v,g,refractory,last,av,ag,control->dt);
}

// metal-deterministic-v2: separate from the frozen serial oracle.
// The three kernels below intentionally retain scalar reference ordering for
// every non-propagation stage.  Only distinct target updates in one source CSR
// row are parallelized.
#include <metal_stdlib>
using namespace metal;

struct StepControl { long clock; uint n; uint slots; uint delay; uint rfc; float dt; };

// Exact target-local counterpart of the frozen serial helper. All mutable
// fields are target-local; source-row target uniqueness therefore makes this
// safe to call concurrently during phase A.
inline void v2_evolve_target(int target, long now, float current,
                             device float *v, device float *g,
                             device short *refractory, device long *last,
                             device const float *av, device const float *ag,
                             float dt) {
  long d=now-last[target]; if(d<=0) return;
  int frozen=refractory[target]>0?refractory[target]-1:0;
  int skip=int(min(d,long(frozen)));
  refractory[target]=d>=refractory[target]?0:short(refractory[target]-d);
  d-=skip;
  if(d>0){
    float a=d<1024?av[d]:exp(-dt*float(d)/20.0f);
    float b=d<1024?ag[d]:exp(-dt*float(d)/5.0f);
    v[target]=-52.0f+(v[target]+52.0f)*a+current*(1.0f-a)+g[target]*(a-b)/3.0f;
    g[target]*=b;
  }
  last[target]=now;
}

// One threadgroup owns all dynamic source ranks for one propagation phase.
// Source rows have verified unique targets. `awakened[e]` is per original edge
// ordinal, so phase B can reproduce original CSR order regardless of lane order.
kernel void v2_drive_settle(
    device float *v [[buffer(0)]], device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]], device float *drive [[buffer(3)]],
    device float *previous [[buffer(4)]], device int *events [[buffer(5)]],
    device int *event_counts [[buffer(6)]], device int *spike_counts [[buffer(7)]],
    device int *active [[buffer(8)]], device uchar *active_flags [[buffer(9)]],
    device int *nactive_ptr [[buffer(10)]], device long *last [[buffer(11)]],
    device const float *av [[buffer(12)]], device const float *ag [[buffer(13)]],
    device StepControl *control [[buffer(14)]], device uchar *changed [[buffer(15)]], uint i [[thread_position_in_grid]]) {
  if(i>=control->n) return;
  const long clock=control->clock;
  if(drive[i]!=previous[i]) {
    v2_evolve_target(int(i),clock-1,previous[i],v,g,refractory,last,av,ag,control->dt);
    previous[i]=drive[i];changed[i]=1;
  } else changed[i]=0;
}

kernel void v2_pre_propagation(
    device float *v [[buffer(0)]], device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]], device float *drive [[buffer(3)]],
    device float *previous [[buffer(4)]], device int *events [[buffer(5)]],
    device int *event_counts [[buffer(6)]], device int *spike_counts [[buffer(7)]],
    device int *active [[buffer(8)]], device uchar *active_flags [[buffer(9)]],
    device int *nactive_ptr [[buffer(10)]], device long *last [[buffer(11)]],
    device const float *av [[buffer(12)]], device const float *ag [[buffer(13)]],
    device StepControl *control [[buffer(14)]], device const uchar *changed [[buffer(15)]], uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  const uint n=control->n, slots=control->slots, delay=control->delay;
  const long clock=control->clock;
  for(uint i=0;i<n;i++) if(changed[i] && !active_flags[i]) {
    active_flags[i]=1; active[nactive_ptr[0]++]=int(i);
  }
  const uint future=uint((clock+long(delay))%long(slots));
  int kept=0, original=nactive_ptr[0];
  for(int k=0;k<original;k++) {
    const int i=active[k];
    v2_evolve_target(i,clock,drive[i],v,g,refractory,last,av,ag,control->dt);
    if(refractory[i]==0 && v[i]>-45.0f) { events[future*n+uint(event_counts[future]++)]=i; spike_counts[i]++; }
    const bool can_fire=v[i]>-45.0f || drive[i]>7.0f || drive[i]+g[i]>7.0f;
    if(can_fire) active[kept++]=i; else active_flags[i]=0;
  }
  nactive_ptr[0]=kept;
}

kernel void v2_ordered_source_propagation(
    device const long *ptr [[buffer(0)]],
    device const int *post [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const int *events [[buffer(3)]], device const int *event_counts [[buffer(4)]],
    device float *v [[buffer(5)]], device float *g [[buffer(6)]],
    device short *refractory [[buffer(7)]], device const float *drive [[buffer(8)]],
    device long *last [[buffer(9)]], device const float *av [[buffer(10)]],
    device const float *ag [[buffer(11)]], device uchar *active_flags [[buffer(12)]],
    device int *active [[buffer(13)]], device int *nactive [[buffer(14)]],
    device uchar *awakened [[buffer(15)]], device StepControl *control [[buffer(16)]],
    uint lane [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]]) {
  const long clock=control->clock; const float dt=control->dt;
  const uint n=control->n, slot=uint(clock%long(control->slots));
  threadgroup int source;
  threadgroup long row_start, row_end;
  const int count=event_counts[slot];
  for(int rank=0;rank<count;rank++) {
    if(lane==0){source=events[slot*n+rank];row_start=ptr[source];row_end=ptr[source+1];}
    threadgroup_barrier(mem_flags::mem_device);
    // Phase A: distinct targets within this source row mean independent writes.
    for(long edge=row_start+lane;edge<row_end;edge+=width) {
      const int target=post[edge];
      v2_evolve_target(target,clock,drive[target],v,g,refractory,last,av,ag,dt);
      const bool accept=refractory[target]==0;
      if(accept) g[target]+=weight[edge];
      awakened[edge]=uchar(accept && active_flags[target]==0);
    }
    threadgroup_barrier(mem_flags::mem_device);
    // Phase B: exact original CSR edge order, never lane or target-ID order.
    if(lane==0) for(long edge=row_start;edge<row_end;edge++) if(awakened[edge]) {
      const int target=post[edge];
      if(!active_flags[target]) { active_flags[target]=1;active[nactive[0]++]=target; }
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

kernel void v2_post_reset(
    device float *v [[buffer(0)]], device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]], device const float *drive [[buffer(3)]],
    device int *events [[buffer(4)]], device int *event_counts [[buffer(5)]],
    device long *last [[buffer(6)]], device const float *av [[buffer(7)]],
    device const float *ag [[buffer(8)]], device StepControl *control [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  if(tid) return;
  const uint n=control->n, slots=control->slots;
  const long clock=control->clock;
  const uint slot=uint(clock%long(slots));
  const uint future=uint((clock+long(control->delay))%long(slots));
  event_counts[slot]=0;
  const int queued=event_counts[future];
  for(int q=0;q<queued;q++) { const int i=events[future*n+uint(q)]; v[i]=-52.0f; g[i]=0.0f; refractory[i]=short(control->rfc); }
}

kernel void v2_materialize_all(device float *v [[buffer(0)]], device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]], device const float *drive [[buffer(3)]],
    device long *last [[buffer(4)]], device const float *av [[buffer(5)]], device const float *ag [[buffer(6)]],
    device StepControl *control [[buffer(7)]], uint i [[thread_position_in_grid]]) {
  if(i>=control->n) return;
  v2_evolve_target(int(i),control->clock,drive[i],v,g,refractory,last,av,ag,control->dt);
}
kernel void v2_finalize(device StepControl *control [[buffer(0)]], uint tid [[thread_position_in_grid]]) { if(!tid) control->clock++; }

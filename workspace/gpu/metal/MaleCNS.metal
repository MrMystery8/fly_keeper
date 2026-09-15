// Experimental fixed-baseline building block. Not selectable as a simulator.
#include <metal_stdlib>
using namespace metal;

struct Params { uint n; float av; float ag; float coupling; float threshold; };

// One thread per neuron. This preserves the CPU's float32 analytic membrane
// equation for an already-materialized, non-refractory state. Queue delivery,
// drive-change settling, and sparse CSR propagation require separate verified
// kernels and are intentionally not approximated here.
kernel void malecns_update(device float *v [[buffer(0)]],
                           device float *g [[buffer(1)]],
                           device const short *refractory [[buffer(2)]],
                           device const float *drive [[buffer(3)]],
                           device int *spike [[buffer(4)]],
                           constant Params &p [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  spike[i]=0;
  if (refractory[i] != 0) return;
  float next=-52.0f+(v[i]+52.0f)*p.av+drive[i]*(1.0f-p.av)+g[i]*p.coupling;
  v[i]=next; g[i]*=p.ag;
  if (next > p.threshold) spike[i]=1;
}

// Deliberately one-thread deterministic reference compaction: preserves active
// list order and does not use atomic append. Performance is irrelevant here.
kernel void stable_compact_serial(device const int *active [[buffer(0)]],
                                  device const int *flags [[buffer(1)]],
                                  device int *ordered [[buffer(2)]],
                                  device uint *count [[buffer(3)]],
                                  constant uint &n [[buffer(4)]],
                                  uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  uint out=0; for(uint k=0;k<n;k++) if(flags[k]) ordered[out++]=active[k];
  count[0]=out;
}

// A single thread intentionally executes the entire reference step. This is
// the golden ordering path: it preserves dynamic queue source order and CSR
// edge order without assuming any cross-threadgroup synchronization.
struct StepControl { long clock; uint n; uint slots; uint delay; uint rfc; float dt; };

inline void evolve_one(int i, long now, float current,
                       device float *v, device float *g, device short *refractory,
                       device long *last, device const float *av, device const float *ag,
                       float dt) {
  long d=now-last[i]; if(d<=0) return;
  int frozen=refractory[i]>0?refractory[i]-1:0;
  int skip=int(min(d,long(frozen)));
  refractory[i]=d>=refractory[i]?0:short(refractory[i]-d); d-=skip;
  if(d>0){
    // Tables are host-generated with the CPU's float32 inputs, matching the
    // native fast path and avoiding a different device exp implementation.
    float a=d<1024?av[d]:exp(-dt*float(d)/20.0f);
    float b=d<1024?ag[d]:exp(-dt*float(d)/5.0f);
    v[i]=-52.0f+(v[i]+52.0f)*a+current*(1.0f-a)+g[i]*(a-b)/3.0f;
    g[i]*=b;
  }
  last[i]=now;
}

kernel void malecns_step_reference_serial(
    device const long *ptr [[buffer(0)]], device const int *post [[buffer(1)]],
    device const float *weight [[buffer(2)]], device float *v [[buffer(3)]],
    device float *g [[buffer(4)]], device short *refractory [[buffer(5)]],
    device float *drive [[buffer(6)]], device float *previous [[buffer(7)]],
    device int *events [[buffer(8)]], device int *event_counts [[buffer(9)]],
    device int *spike_counts [[buffer(10)]], device int *active [[buffer(11)]],
    device uchar *active_flags [[buffer(12)]], device int *nactive_ptr [[buffer(13)]],
    device long *last [[buffer(14)]], device const float *av [[buffer(15)]],
    device const float *ag [[buffer(16)]], device StepControl *control [[buffer(17)]],
    uint tid [[thread_position_in_grid]]) {
  if(tid) return;
  const uint n=control->n,slots=control->slots,delay=control->delay;
  const long clock=control->clock;
  // Drive changes settle old current in ascending internal-index order.
  for(uint i=0;i<n;i++) if(drive[i]!=previous[i]){
    evolve_one(int(i),clock-1,previous[i],v,g,refractory,last,av,ag,control->dt);
    previous[i]=drive[i];
    if(!active_flags[i]){active_flags[i]=1;active[nactive_ptr[0]++]=int(i);}
  }
  const uint slot=uint(clock%long(slots));
  const uint future=uint((clock+long(delay))%long(slots));
  int kept=0,original=nactive_ptr[0];
  for(int k=0;k<original;k++){
    const int i=active[k]; evolve_one(i,clock,drive[i],v,g,refractory,last,av,ag,control->dt);
    if(refractory[i]==0&&v[i]>-45.0f){events[future*n+uint(event_counts[future]++)]=i;spike_counts[i]++;}
    const bool can_fire=v[i]>-45.0f||drive[i]>7.0f||drive[i]+g[i]>7.0f;
    if(can_fire)active[kept++]=i;else active_flags[i]=0;
  }
  nactive_ptr[0]=kept;
  // Dynamic queue order then original source-major CSR order.
  const int delivered=event_counts[slot];
  for(int q=0;q<delivered;q++){
    const int source=events[slot*n+uint(q)];
    for(long e=ptr[source];e<ptr[source+1];e++){
      const int target=post[e]; evolve_one(target,clock,drive[target],v,g,refractory,last,av,ag,control->dt);
      if(refractory[target]==0){g[target]+=weight[e];if(!active_flags[target]){active_flags[target]=1;active[nactive_ptr[0]++]=target;}}
    }
  }
  event_counts[slot]=0;
  const int queued=event_counts[future];
  for(int q=0;q<queued;q++){const int i=events[future*n+uint(q)];v[i]=-52.0f;g[i]=0.0f;refractory[i]=short(control->rfc);}
  // Native calls this observation-boundary materialization after each public
  // one-step invocation. The reference bridge also dispatches one logical
  // step per call, so reproduce it here for complete state comparison.
  for(uint i=0;i<n;i++) evolve_one(int(i),clock,drive[i],v,g,refractory,last,av,ag,control->dt);
  control->clock=clock+1;
}

// Throughput-oriented GPU-resident backend.  Unlike deterministic v4, this
// uses dense parallel neuron evolution and atomic sparse propagation.  It is
// deterministic at the spike/readout level only when validated empirically;
// floating-point accumulation order is intentionally relaxed.
#include <metal_stdlib>
using namespace metal;

struct V5Control { long clock; uint n,slots,delay,rfc; float dt; };
struct DispatchArgs { uint x,y,z; };

inline void v5_evolve(int i,long now,float current,device float*v,device float*g,
 device short*r,device long*last,device const float*av,device const float*ag,float dt){
 long d=now-last[i];if(d<=0)return;int frozen=r[i]>0?r[i]-1:0;int skip=int(min(d,long(frozen)));
 r[i]=d>=r[i]?0:short(r[i]-d);d-=skip;if(d>0){float a=d<1024?av[d]:exp(-dt*float(d)/20.f),b=d<1024?ag[d]:exp(-dt*float(d)/5.f);v[i]=-52.f+(v[i]+52.f)*a+current*(1.f-a)+g[i]*(a-b)/3.f;g[i]*=b;}last[i]=now;
}
kernel void v5_update(device float*v[[buffer(0)]],device float*g[[buffer(1)]],device short*r[[buffer(2)]],
 device const float*drive[[buffer(3)]],device float*previous[[buffer(4)]],device int*events[[buffer(5)]],
 device atomic_int*event_counts[[buffer(6)]],device int*spike_counts[[buffer(7)]],device atomic_uint*flags[[buffer(8)]],
 device long*last[[buffer(9)]],device const float*av[[buffer(10)]],device const float*ag[[buffer(11)]],
 device const V5Control*c[[buffer(12)]],device uchar*pending_reset[[buffer(13)]],constant long&clock[[buffer(14)]],device DispatchArgs*propagate_args[[buffer(15)]],uint i[[thread_position_in_grid]]){
 if(i==0){uint slot=uint(clock%long(c->slots));uint sources=uint(max(0,atomic_load_explicit(event_counts+slot,memory_order_relaxed)));propagate_args[0]={sources,1,1};atomic_store_explicit(event_counts+slot,0,memory_order_relaxed);}
 if(i>=c->n)return;
 // Reset is deferred from the end of the preceding tick to the beginning of
 // this one. No state can observe the interval between those kernel phases.
 if(pending_reset[i]){v[i]=-52.f;g[i]=0.f;r[i]=short(c->rfc);pending_reset[i]=0;}
 float d=drive[i];if(d!=previous[i]){previous[i]=d;atomic_store_explicit(flags+i,1,memory_order_relaxed);}
 v5_evolve(int(i),clock,d,v,g,r,last,av,ag,c->dt);if(!atomic_load_explicit(flags+i,memory_order_relaxed))return;
 if(r[i]==0&&v[i]>-45.f){uint future=uint((clock+long(c->delay))%long(c->slots));int q=atomic_fetch_add_explicit(event_counts+future,1,memory_order_relaxed);events[future*c->n+uint(q)]=int(i);spike_counts[i]++;pending_reset[i]=1;}
 bool keep=v[i]>-45.f||d>7.f||d+g[i]>7.f;atomic_store_explicit(flags+i,keep?1u:0u,memory_order_relaxed);
}

kernel void v5_prepare(device const atomic_int*event_counts[[buffer(0)]],device const V5Control*c[[buffer(1)]],
 device DispatchArgs*propagate[[buffer(2)]],uint tid[[thread_position_in_grid]]){
 if(tid)return;uint slot=uint(c->clock%long(c->slots));uint sources=uint(max(0,atomic_load_explicit(event_counts+slot,memory_order_relaxed)));propagate[0]={sources,1,1};
}

kernel void v5_propagate(device const long*ptr[[buffer(0)]],device const int*post[[buffer(1)]],device const float*weight[[buffer(2)]],
 device const int*events[[buffer(3)]],device atomic_float*g[[buffer(4)]],device atomic_uint*flags[[buffer(5)]],device const V5Control*c[[buffer(6)]],
 constant long&clock[[buffer(7)]],uint lane[[thread_index_in_threadgroup]],uint rank[[threadgroup_position_in_grid]],uint width[[threads_per_threadgroup]]){
 uint slot=uint(clock%long(c->slots));int source=events[slot*c->n+rank];long begin=ptr[source],end=ptr[source+1];
 for(long e=begin+long(lane);e<end;e+=long(width)){int target=post[e];atomic_fetch_add_explicit(g+target,weight[e],memory_order_relaxed);atomic_store_explicit(flags+target,1,memory_order_relaxed);}
}

kernel void v5_reset(device float*v[[buffer(0)]],device atomic_float*g[[buffer(1)]],device short*r[[buffer(2)]],
 device const int*events[[buffer(3)]],device const atomic_int*event_counts[[buffer(4)]],device const V5Control*c[[buffer(5)]],uint q[[thread_position_in_grid]]){
 uint future=uint((c->clock+long(c->delay))%long(c->slots));uint count=uint(max(0,atomic_load_explicit(event_counts+future,memory_order_relaxed)));if(q>=count)return;
 int i=events[future*c->n+q];v[i]=-52.f;atomic_store_explicit(g+i,0.f,memory_order_relaxed);r[i]=short(c->rfc);
}

kernel void v5_finalize(device atomic_int*event_counts[[buffer(0)]],device V5Control*c[[buffer(1)]],device DispatchArgs*next[[buffer(2)]],uint tid[[thread_position_in_grid]]){
 if(tid)return;uint slot=uint(c->clock%long(c->slots));atomic_store_explicit(event_counts+slot,0,memory_order_relaxed);c->clock++;uint next_slot=uint(c->clock%long(c->slots));uint sources=uint(max(0,atomic_load_explicit(event_counts+next_slot,memory_order_relaxed)));next[0]={sources,1,1};
}
kernel void v5_apply_pending_resets(device float*v[[buffer(0)]],device float*g[[buffer(1)]],device short*r[[buffer(2)]],device uchar*pending[[buffer(3)]],device const V5Control*c[[buffer(4)]],device long*last[[buffer(5)]],device const float*drive[[buffer(6)]],device const float*av[[buffer(7)]],device const float*ag[[buffer(8)]],constant long&boundary[[buffer(9)]],uint i[[thread_position_in_grid]]){if(i>=c->n)return;if(pending[i]){v[i]=-52.f;g[i]=0.f;r[i]=short(c->rfc);pending[i]=0;}v5_evolve(int(i),boundary,drive[i],v,g,r,last,av,ag,c->dt);}

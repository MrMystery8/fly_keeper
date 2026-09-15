// Persistent bridge for metal-deterministic-v2.  It deliberately does not
// share mutable code or binaries with the frozen serial bridge.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstring>

struct StepControl { int64_t clock; uint32_t n, slots, delay, rfc; float dt; };
struct V2Context {
  id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLLibrary> library;
  id<MTLComputePipelineState> settle, pre, propagate, post, materialize, finalize;
  id<MTLBuffer> ptr, postIds, weight, v, g, refr, last, drive, previous;
  id<MTLBuffer> events, eventCounts, spikeCounts, active, flags, nactive, awakened;
  id<MTLBuffer> av, ag, control, changed;
  NSUInteger n, e, width;
};

static void encode(V2Context *c, id<MTLCommandBuffer> cb, id<MTLComputePipelineState> p,
                   NSArray<id<MTLBuffer>> *buffers, NSUInteger threads, NSUInteger tpg) {
  id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder]; [enc setComputePipelineState:p];
  for(NSUInteger i=0;i<buffers.count;i++) [enc setBuffer:buffers[i] offset:0 atIndex:i];
  [enc dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)]; [enc endEncoding];
}
extern "C" {
void *metal_create_deterministic_v2(int n,long long e,const long long *ptr,const int *post,const float *weight,const char *libraryPath,int width) {
 @autoreleasepool {
  V2Context *c=new V2Context{}; c->n=n;c->e=e;c->device=MTLCreateSystemDefaultDevice(); if(!c->device){delete c;return nullptr;}
  c->width=width>0?NSUInteger(width):64; c->width=MIN(c->width,c->device.maxThreadsPerThreadgroup.width); c->queue=[c->device newCommandQueue];
  NSError *err=nil;c->library=[c->device newLibraryWithFile:[NSString stringWithUTF8String:libraryPath] error:&err];if(!c->library){delete c;return nullptr;}
  auto pipe=[&](NSString *name){return [c->device newComputePipelineStateWithFunction:[c->library newFunctionWithName:name] error:&err];};
  c->settle=pipe(@"v2_drive_settle");c->pre=pipe(@"v2_pre_propagation");c->propagate=pipe(@"v2_ordered_source_propagation");c->post=pipe(@"v2_post_reset");c->materialize=pipe(@"v2_materialize_all");c->finalize=pipe(@"v2_finalize");if(!c->settle||!c->pre||!c->propagate||!c->post||!c->materialize||!c->finalize){delete c;return nullptr;}
  auto copy=[&](const void *p,NSUInteger z){return [c->device newBufferWithBytes:p length:z options:MTLResourceStorageModeShared];};auto zero=[&](NSUInteger z){return [c->device newBufferWithLength:z options:MTLResourceStorageModeShared];};
  c->ptr=copy(ptr,(n+1)*sizeof(int64_t));c->postIds=copy(post,e*sizeof(int32_t));c->weight=copy(weight,e*sizeof(float));c->v=zero(n*4);c->g=zero(n*4);c->refr=zero(n*2);c->last=zero(n*8);c->drive=zero(n*4);c->previous=zero(n*4);c->events=zero(19ull*n*4);c->eventCounts=zero(19*4);c->spikeCounts=zero(n*4);c->active=zero(n*4);c->flags=zero(n);c->nactive=zero(4);c->awakened=zero(e);c->changed=zero(n);
  float av[1024],ag[1024];for(int i=0;i<1024;i++){av[i]=std::exp(-.1f*i/20.f);ag[i]=std::exp(-.1f*i/5.f);}c->av=copy(av,sizeof(av));c->ag=copy(ag,sizeof(ag));StepControl ctl{0,uint32_t(n),19,18,22,.1f};c->control=copy(&ctl,sizeof(ctl));return c;
 }
}
void metal_destroy_v2(void *p){if(p)delete static_cast<V2Context*>(p);}
int metal_load_state_v2(void *p,const float*v,const float*g,const int16_t*r,const float*d,const float*previous,const int32_t*events,const int32_t*eventCounts,const int32_t*spikeCounts,const int32_t*active,const uint8_t*flags,const int32_t*nactive,const int64_t*last,const int64_t*clock){if(!p)return 0;auto*c=static_cast<V2Context*>(p);NSUInteger n=c->n;memcpy(c->v.contents,v,n*4);memcpy(c->g.contents,g,n*4);memcpy(c->refr.contents,r,n*2);memcpy(c->drive.contents,d,n*4);memcpy(c->previous.contents,previous,n*4);memcpy(c->events.contents,events,19*n*4);memcpy(c->eventCounts.contents,eventCounts,19*4);memcpy(c->spikeCounts.contents,spikeCounts,n*4);memcpy(c->active.contents,active,n*4);memcpy(c->flags.contents,flags,n);memcpy(c->nactive.contents,nactive,4);memcpy(c->last.contents,last,n*8);static_cast<StepControl*>(c->control.contents)->clock=*clock;return 1;}
int metal_copy_state_v2(void *p,float*v,float*g,int16_t*r,float*d,float*previous,int32_t*events,int32_t*eventCounts,int32_t*spikeCounts,int32_t*active,uint8_t*flags,int32_t*nactive,int64_t*last,int64_t*clock){if(!p)return 0;auto*c=static_cast<V2Context*>(p);NSUInteger n=c->n;memcpy(v,c->v.contents,n*4);memcpy(g,c->g.contents,n*4);memcpy(r,c->refr.contents,n*2);memcpy(d,c->drive.contents,n*4);memcpy(previous,c->previous.contents,n*4);memcpy(events,c->events.contents,19*n*4);memcpy(eventCounts,c->eventCounts.contents,19*4);memcpy(spikeCounts,c->spikeCounts.contents,n*4);memcpy(active,c->active.contents,n*4);memcpy(flags,c->flags.contents,n);memcpy(nactive,c->nactive.contents,4);memcpy(last,c->last.contents,n*8);*clock=static_cast<StepControl*>(c->control.contents)->clock;return 1;}
int metal_set_drive_v2(void*p,const float*d){if(!p)return 0;auto*c=static_cast<V2Context*>(p);memcpy(c->drive.contents,d,c->n*4);return 1;}
int metal_zero_spike_counts_v2(void*p){if(!p)return 0;auto*c=static_cast<V2Context*>(p);memset(c->spikeCounts.contents,0,c->n*4);return 1;}
int metal_step_v2_batch(void*p,int steps){if(!p||steps<1)return 0;auto*c=static_cast<V2Context*>(p);id<MTLCommandBuffer>cb=[c->queue commandBuffer];for(int tick=0;tick<steps;tick++){encode(c,cb,c->settle,@[c->v,c->g,c->refr,c->drive,c->previous,c->events,c->eventCounts,c->spikeCounts,c->active,c->flags,c->nactive,c->last,c->av,c->ag,c->control,c->changed],c->n,c->width);encode(c,cb,c->pre,@[c->v,c->g,c->refr,c->drive,c->previous,c->events,c->eventCounts,c->spikeCounts,c->active,c->flags,c->nactive,c->last,c->av,c->ag,c->control,c->changed],1,1);encode(c,cb,c->propagate,@[c->ptr,c->postIds,c->weight,c->events,c->eventCounts,c->v,c->g,c->refr,c->drive,c->last,c->av,c->ag,c->flags,c->active,c->nactive,c->awakened,c->control],c->width,c->width);encode(c,cb,c->post,@[c->v,c->g,c->refr,c->drive,c->events,c->eventCounts,c->last,c->av,c->ag,c->control],1,1);encode(c,cb,c->materialize,@[c->v,c->g,c->refr,c->drive,c->last,c->av,c->ag,c->control],c->n,c->width);encode(c,cb,c->finalize,@[c->control],1,1);}[cb commit];[cb waitUntilCompleted];return cb.status==MTLCommandBufferStatusCompleted;}
int metal_step_v2(void*p){return metal_step_v2_batch(p,1);}
}

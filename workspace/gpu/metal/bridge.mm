// Persistent native Metal reference bridge.  CPU remains the only supported
// simulator; this exposes narrow test hooks for parity work.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstring>

struct StepControl { int64_t clock; uint32_t n, slots, delay, rfc; float dt; };
struct MetalContext {
  id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLLibrary> library;
  id<MTLComputePipelineState> update, compact, step;
  id<MTLBuffer> ptr, post, weight, v, g, refr, last, drive, previous;
  id<MTLBuffer> events, eventCounts, spikeCounts, active, flags, nactive;
  id<MTLBuffer> av, ag, control;
  NSUInteger n, e;
};

static void run(MetalContext *c, id<MTLComputePipelineState> p,
                NSArray<id<MTLBuffer>> *buffers) {
  id<MTLCommandBuffer> cb=[c->queue commandBuffer];
  id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
  [enc setComputePipelineState:p];
  for (NSUInteger i=0;i<buffers.count;i++) [enc setBuffer:buffers[i] offset:0 atIndex:i];
  [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
}

extern "C" {
void *metal_create_deterministic_v1(int n, long long e, const long long *ptr,
                                    const int *post, const float *weight,
                                    const char *libraryPath) {
 @autoreleasepool {
  MetalContext *c=new MetalContext{}; c->n=n;c->e=e;c->device=MTLCreateSystemDefaultDevice();
  if(!c->device){delete c;return nullptr;} c->queue=[c->device newCommandQueue];
  NSError *err=nil; c->library=[c->device newLibraryWithFile:[NSString stringWithUTF8String:libraryPath] error:&err];
  if(!c->library){delete c;return nullptr;}
  auto pipeline=[&](NSString *name)->id<MTLComputePipelineState>{return [c->device newComputePipelineStateWithFunction:[c->library newFunctionWithName:name] error:&err];};
  c->update=pipeline(@"malecns_update"); c->compact=pipeline(@"stable_compact_serial"); c->step=pipeline(@"malecns_step_reference_serial");
  if(!c->update||!c->compact||!c->step){delete c;return nullptr;}
  auto copy=[&](const void *p,NSUInteger bytes){return [c->device newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];};
  auto zero=[&](NSUInteger bytes){return [c->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];};
  c->ptr=copy(ptr,(n+1)*sizeof(int64_t));c->post=copy(post,e*sizeof(int32_t));c->weight=copy(weight,e*sizeof(float));
  c->v=zero(n*4);c->g=zero(n*4);c->refr=zero(n*2);c->last=zero(n*8);c->drive=zero(n*4);c->previous=zero(n*4);
  c->events=zero(19ull*n*4);c->eventCounts=zero(19*4);c->spikeCounts=zero(n*4);c->active=zero(n*4);c->flags=zero(n);c->nactive=zero(4);
  float av[1024],ag[1024];for(int i=0;i<1024;i++){av[i]=std::exp(-.1f*i/20.f);ag[i]=std::exp(-.1f*i/5.f);}c->av=copy(av,sizeof(av));c->ag=copy(ag,sizeof(ag));
  StepControl ctl{-0,static_cast<uint32_t>(n),19,18,22,.1f}; c->control=copy(&ctl,sizeof(ctl));return c;
 }
}
void metal_destroy(void *p){if(p)delete static_cast<MetalContext*>(p);}

// Copies a complete tiny/reference state into persistent shared buffers.
int metal_load_state(void *p,const float *v,const float *g,const int16_t *refr,const float *drive,const float *previous,
                     const int32_t *events,const int32_t *eventCounts,const int32_t *spikeCounts,const int32_t *active,
                     const uint8_t *flags,const int32_t *nactive,const int64_t *last,const int64_t *clock){
  if(!p)return 0;auto*c=static_cast<MetalContext*>(p);const NSUInteger n=c->n;
  memcpy(c->v.contents,v,n*4);memcpy(c->g.contents,g,n*4);memcpy(c->refr.contents,refr,n*2);memcpy(c->drive.contents,drive,n*4);memcpy(c->previous.contents,previous,n*4);
  memcpy(c->events.contents,events,19*n*4);memcpy(c->eventCounts.contents,eventCounts,19*4);memcpy(c->spikeCounts.contents,spikeCounts,n*4);memcpy(c->active.contents,active,n*4);memcpy(c->flags.contents,flags,n);memcpy(c->nactive.contents,nactive,4);memcpy(c->last.contents,last,n*8);
  static_cast<StepControl*>(c->control.contents)->clock=*clock;return 1;
}
int metal_copy_state(void *p,float *v,float *g,int16_t *refr,float *drive,float *previous,int32_t *events,int32_t *eventCounts,
                     int32_t *spikeCounts,int32_t *active,uint8_t *flags,int32_t *nactive,int64_t *last,int64_t *clock){
  if(!p)return 0;auto*c=static_cast<MetalContext*>(p);const NSUInteger n=c->n;
  memcpy(v,c->v.contents,n*4);memcpy(g,c->g.contents,n*4);memcpy(refr,c->refr.contents,n*2);memcpy(drive,c->drive.contents,n*4);memcpy(previous,c->previous.contents,n*4);memcpy(events,c->events.contents,19*n*4);memcpy(eventCounts,c->eventCounts.contents,19*4);memcpy(spikeCounts,c->spikeCounts.contents,n*4);memcpy(active,c->active.contents,n*4);memcpy(flags,c->flags.contents,n);memcpy(nactive,c->nactive.contents,4);memcpy(last,c->last.contents,n*8);*clock=static_cast<StepControl*>(c->control.contents)->clock;return 1;
}
int metal_step_reference_serial(void *p){
  if(!p)return 0;auto*c=static_cast<MetalContext*>(p);
  run(c,c->step,@[c->ptr,c->post,c->weight,c->v,c->g,c->refr,c->drive,c->previous,c->events,c->eventCounts,c->spikeCounts,c->active,c->flags,c->nactive,c->last,c->av,c->ag,c->control]);return 1;
}
int metal_set_drive(void *p,const float *drive){
  if(!p)return 0;auto*c=static_cast<MetalContext*>(p);memcpy(c->drive.contents,drive,c->n*sizeof(float));return 1;
}
int metal_zero_spike_counts(void *p){
  if(!p)return 0;auto*c=static_cast<MetalContext*>(p);memset(c->spikeCounts.contents,0,c->n*sizeof(int32_t));return 1;
}
int metal_compact_test(void *p,const int32_t *active,const int32_t *flags,int n,int32_t *ordered,uint32_t *count){
  if(!p||n<0)return 0;if(n==0){*count=0;return 1;}auto*c=static_cast<MetalContext*>(p);auto mk=[&](const void*x,NSUInteger z){return [c->device newBufferWithBytes:x length:z options:MTLResourceStorageModeShared];};
  id<MTLBuffer>a=mk(active,n*4),f=mk(flags,n*4),o=[c->device newBufferWithLength:n*4 options:MTLResourceStorageModeShared],q=[c->device newBufferWithLength:4 options:MTLResourceStorageModeShared],nn=mk(&n,4);
  run(c,c->compact,@[a,f,o,q,nn]);memcpy(ordered,o.contents,n*4);memcpy(count,q.contents,4);return 1;
}
}

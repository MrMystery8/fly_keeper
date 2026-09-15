// Persistent bridge for the batched deterministic v4 engine.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstring>

struct V4Control {
  int64_t clock; uint32_t n,slots,delay,rfc; float dt;
  uint64_t total_sources,total_edges,total_active_visits;
  uint32_t max_sources,max_edges,max_active;
};
struct V4Context {
  id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLLibrary> library;
  id<MTLComputePipelineState> step;
  id<MTLBuffer> ptr,postIds,weight,v,g,refr,last,drive,previous,events,eventCounts;
  id<MTLBuffer> spikeCounts,active,flags,nactive,av,ag,changed,control;
  NSUInteger n,e,width;
};

extern "C" {
void *metal_create_deterministic_v4(int n,long long e,const long long *ptr,const int *post,
                                    const float *weight,const char *library_path,int width) {
 @autoreleasepool {
  auto*c=new V4Context{}; c->n=n;c->e=e;c->device=MTLCreateSystemDefaultDevice();
  if(!c->device){delete c;return nullptr;} c->queue=[c->device newCommandQueue];
  c->width=width>0?NSUInteger(width):256; c->width=MIN(c->width,c->device.maxThreadsPerThreadgroup.width);
  NSError*err=nil;c->library=[c->device newLibraryWithFile:[NSString stringWithUTF8String:library_path] error:&err];
  if(!c->library){delete c;return nullptr;}
  c->step=[c->device newComputePipelineStateWithFunction:[c->library newFunctionWithName:@"v4_step_batch"] error:&err];
  if(!c->step){delete c;return nullptr;}
  auto copy=[&](const void*p,NSUInteger z){return[c->device newBufferWithBytes:p length:z options:MTLResourceStorageModeShared];};
  auto zero=[&](NSUInteger z){return[c->device newBufferWithLength:z options:MTLResourceStorageModeShared];};
  c->ptr=copy(ptr,(n+1)*8);c->postIds=copy(post,e*4);c->weight=copy(weight,e*4);
  c->v=zero(n*4);c->g=zero(n*4);c->refr=zero(n*2);c->last=zero(n*8);c->drive=zero(n*4);c->previous=zero(n*4);
  c->events=zero(19ull*n*4);c->eventCounts=zero(19*4);c->spikeCounts=zero(n*4);c->active=zero(n*4);c->flags=zero(n);c->nactive=zero(4);c->changed=zero(n);
  float av[1024],ag[1024];for(int i=0;i<1024;i++){av[i]=std::exp(-.1f*i/20.f);ag[i]=std::exp(-.1f*i/5.f);}
  c->av=copy(av,sizeof(av));c->ag=copy(ag,sizeof(ag));V4Control ctl{0,uint32_t(n),19,18,22,.1f,0,0,0,0,0,0};c->control=copy(&ctl,sizeof(ctl));
  if(!c->ptr||!c->postIds||!c->weight||!c->v||!c->g||!c->events||!c->control){delete c;return nullptr;} return c;
 }
}
void metal_destroy_v4(void*p){if(p)delete static_cast<V4Context*>(p);}
int metal_load_state_v4(void*p,const float*v,const float*g,const int16_t*r,const float*d,const float*previous,
 const int32_t*events,const int32_t*eventCounts,const int32_t*spikeCounts,const int32_t*active,const uint8_t*flags,
 const int32_t*nactive,const int64_t*last,const int64_t*clock){
 if(!p)return 0;auto*c=static_cast<V4Context*>(p);NSUInteger n=c->n;
 memcpy(c->v.contents,v,n*4);memcpy(c->g.contents,g,n*4);memcpy(c->refr.contents,r,n*2);memcpy(c->drive.contents,d,n*4);memcpy(c->previous.contents,previous,n*4);
 memcpy(c->events.contents,events,19*n*4);memcpy(c->eventCounts.contents,eventCounts,19*4);memcpy(c->spikeCounts.contents,spikeCounts,n*4);memcpy(c->active.contents,active,n*4);memcpy(c->flags.contents,flags,n);memcpy(c->nactive.contents,nactive,4);memcpy(c->last.contents,last,n*8);static_cast<V4Control*>(c->control.contents)->clock=*clock;return 1;
}
int metal_copy_state_v4(void*p,float*v,float*g,int16_t*r,float*d,float*previous,int32_t*events,int32_t*eventCounts,
 int32_t*spikeCounts,int32_t*active,uint8_t*flags,int32_t*nactive,int64_t*last,int64_t*clock){
 if(!p)return 0;auto*c=static_cast<V4Context*>(p);NSUInteger n=c->n;
 memcpy(v,c->v.contents,n*4);memcpy(g,c->g.contents,n*4);memcpy(r,c->refr.contents,n*2);memcpy(d,c->drive.contents,n*4);memcpy(previous,c->previous.contents,n*4);memcpy(events,c->events.contents,19*n*4);memcpy(eventCounts,c->eventCounts.contents,19*4);memcpy(spikeCounts,c->spikeCounts.contents,n*4);memcpy(active,c->active.contents,n*4);memcpy(flags,c->flags.contents,n);memcpy(nactive,c->nactive.contents,4);memcpy(last,c->last.contents,n*8);*clock=static_cast<V4Control*>(c->control.contents)->clock;return 1;
}
int metal_set_drive_v4(void*p,const float*d){if(!p)return 0;auto*c=static_cast<V4Context*>(p);memcpy(c->drive.contents,d,c->n*4);return 1;}
int metal_zero_spike_counts_v4(void*p){if(!p)return 0;auto*c=static_cast<V4Context*>(p);memset(c->spikeCounts.contents,0,c->n*4);return 1;}
int metal_read_v4(void*p,const int32_t*indices,uint32_t count,int32_t*counts,float*voltages){if(!p)return 0;auto*c=static_cast<V4Context*>(p);auto*sc=(int32_t*)c->spikeCounts.contents;auto*vv=(float*)c->v.contents;for(uint32_t i=0;i<count;i++){if(indices[i]<0||NSUInteger(indices[i])>=c->n)return 0;counts[i]=sc[indices[i]];voltages[i]=vv[indices[i]];}return 1;}
int metal_stats_v4(void*p,V4Control*out){if(!p||!out)return 0;memcpy(out,static_cast<V4Context*>(p)->control.contents,sizeof(V4Control));return 1;}
int metal_step_v4_batch(void*p,int steps){
 @autoreleasepool {if(!p||steps<1)return 0;auto*c=static_cast<V4Context*>(p);uint32_t count=uint32_t(steps);
  id<MTLCommandBuffer>cb=[c->queue commandBuffer];id<MTLComputeCommandEncoder>x=[cb computeCommandEncoder];[x setComputePipelineState:c->step];
  NSArray<id<MTLBuffer>>*b=@[c->ptr,c->postIds,c->weight,c->v,c->g,c->refr,c->drive,c->previous,c->events,c->eventCounts,c->spikeCounts,c->active,c->flags,c->nactive,c->last,c->av,c->ag,c->changed,c->control];
  for(NSUInteger i=0;i<b.count;i++)[x setBuffer:b[i] offset:0 atIndex:i];[x setBytes:&count length:sizeof(count) atIndex:19];
  [x dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(c->width,1,1)];[x endEncoding];[cb commit];[cb waitUntilCompleted];return cb.status==MTLCommandBufferStatusCompleted;
 }
}
int metal_step_v4(void*p){return metal_step_v4_batch(p,1);}
}

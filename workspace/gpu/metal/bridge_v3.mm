// Independent primitive bridge for deterministic-v3.  This intentionally owns
// no frozen-oracle mutable code and is expanded into the persistent step bridge
// only after its worklist primitives are proven.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

struct RadixControl { uint32_t count, blocks, shift; };
struct V3PrimitiveContext {
  id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLLibrary> library;
  id<MTLComputePipelineState> hist, prefix, scatter, mark, compact, stable, source_prefix, expand;
};

extern "C" {
void *metal_v3_primitive_create(const char *library_path) {
 @autoreleasepool {
  auto *c = new V3PrimitiveContext{}; c->device=MTLCreateSystemDefaultDevice();
  if(!c->device) { delete c; return nullptr; }
  c->queue=[c->device newCommandQueue]; NSError *error=nil;
  c->library=[c->device newLibraryWithFile:[NSString stringWithUTF8String:library_path] error:&error];
  if(!c->library) { delete c; return nullptr; }
  auto pipeline=[&](NSString *name) { return [c->device newComputePipelineStateWithFunction:[c->library newFunctionWithName:name] error:&error]; };
  c->hist=pipeline(@"v3_radix_hist"); c->prefix=pipeline(@"v3_radix_prefix"); c->scatter=pipeline(@"v3_radix_scatter"); c->mark=pipeline(@"v3_mark_segments"); c->compact=pipeline(@"v3_compact_segments"); c->stable=pipeline(@"v3_stable_compact");c->source_prefix=pipeline(@"v3_source_prefix");c->expand=pipeline(@"v3_expand_edges");
  if(!c->hist || !c->prefix || !c->scatter || !c->mark || !c->compact || !c->stable || !c->source_prefix || !c->expand) { delete c; return nullptr; }
  return c;
 }
}
int metal_v3_segment64(void *p,const uint64_t *keys,uint32_t count,uint32_t *targets,uint32_t *starts,uint32_t *ends,uint32_t *out_count) {
 @autoreleasepool { if(!p) return 0; if(!count){*out_count=0;return 1;} auto*c=static_cast<V3PrimitiveContext*>(p); auto sh=MTLResourceStorageModeShared;
  id<MTLBuffer> k=[c->device newBufferWithBytes:keys length:NSUInteger(count)*8 options:sh], h=[c->device newBufferWithLength:count options:sh],t=[c->device newBufferWithLength:NSUInteger(count)*4 options:sh],s=[c->device newBufferWithLength:NSUInteger(count)*4 options:sh],e=[c->device newBufferWithLength:NSUInteger(count)*4 options:sh],n=[c->device newBufferWithLength:4 options:sh];
  struct {uint32_t n,slot,source_count,edge_count,capacity;} ctl{0,0,0,count,count}; id<MTLBuffer> control=[c->device newBufferWithBytes:&ctl length:sizeof(ctl) options:sh]; id<MTLCommandBuffer> cb=[c->queue commandBuffer]; id<MTLComputeCommandEncoder>x=[cb computeCommandEncoder];[x setComputePipelineState:c->mark];[x setBuffer:k offset:0 atIndex:0];[x setBuffer:h offset:0 atIndex:1];[x setBuffer:control offset:0 atIndex:2];[x dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(count,NSUInteger(256)),1,1)];[x endEncoding];x=[cb computeCommandEncoder];[x setComputePipelineState:c->compact];[x setBuffer:k offset:0 atIndex:0];[x setBuffer:h offset:0 atIndex:1];[x setBuffer:t offset:0 atIndex:2];[x setBuffer:s offset:0 atIndex:3];[x setBuffer:e offset:0 atIndex:4];[x setBuffer:n offset:0 atIndex:5];[x setBuffer:control offset:0 atIndex:6];[x dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[x endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;*out_count=*static_cast<uint32_t*>(n.contents);memcpy(targets,t.contents,*out_count*4);memcpy(starts,s.contents,*out_count*4);memcpy(ends,e.contents,*out_count*4);return 1; }
}
void metal_v3_primitive_destroy(void *p) { if(p) delete static_cast<V3PrimitiveContext *>(p); }
int metal_v3_sort64(void *p, const uint64_t *keys, const uint32_t *payload, uint32_t count,
                    uint64_t *out_keys, uint32_t *out_payload) {
 @autoreleasepool {
  if(!p || (!keys && count) || (!payload && count)) return 0;
  if(!count) return 1;
  auto *c=static_cast<V3PrimitiveContext *>(p); const uint32_t blocks=(count+255)/256;
  auto shared=MTLResourceStorageModeShared;
  id<MTLBuffer> key_a=[c->device newBufferWithBytes:keys length:NSUInteger(count)*sizeof(uint64_t) options:shared];
  id<MTLBuffer> key_b=[c->device newBufferWithLength:NSUInteger(count)*sizeof(uint64_t) options:shared];
  id<MTLBuffer> pay_a=[c->device newBufferWithBytes:payload length:NSUInteger(count)*sizeof(uint32_t) options:shared];
  id<MTLBuffer> pay_b=[c->device newBufferWithLength:NSUInteger(count)*sizeof(uint32_t) options:shared];
  id<MTLBuffer> histogram=[c->device newBufferWithLength:NSUInteger(blocks)*16*sizeof(uint32_t) options:shared];
  id<MTLBuffer> offsets=[c->device newBufferWithLength:NSUInteger(blocks)*16*sizeof(uint32_t) options:shared];
  if(!key_a || !key_b || !pay_a || !pay_b || !histogram || !offsets) return 0;
  id<MTLCommandBuffer> cb=[c->queue commandBuffer];
  for(uint32_t shift=0; shift<64; shift+=4) {
    RadixControl control{count,blocks,shift};
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder]; [e setComputePipelineState:c->hist];
    [e setBuffer:key_a offset:0 atIndex:0]; [e setBuffer:histogram offset:0 atIndex:1]; [e setBytes:&control length:sizeof(control) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(blocks,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e endEncoding];
    e=[cb computeCommandEncoder]; [e setComputePipelineState:c->prefix]; [e setBuffer:histogram offset:0 atIndex:0]; [e setBuffer:offsets offset:0 atIndex:1]; [e setBytes:&control length:sizeof(control) atIndex:2]; [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e endEncoding];
    e=[cb computeCommandEncoder]; [e setComputePipelineState:c->scatter]; [e setBuffer:key_a offset:0 atIndex:0]; [e setBuffer:pay_a offset:0 atIndex:1]; [e setBuffer:key_b offset:0 atIndex:2]; [e setBuffer:pay_b offset:0 atIndex:3]; [e setBuffer:offsets offset:0 atIndex:4]; [e setBytes:&control length:sizeof(control) atIndex:5]; [e dispatchThreadgroups:MTLSizeMake(blocks,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e endEncoding];
    auto tk=key_a; key_a=key_b; key_b=tk; auto tp=pay_a; pay_a=pay_b; pay_b=tp;
  }
  [cb commit]; [cb waitUntilCompleted]; if(cb.status!=MTLCommandBufferStatusCompleted) return 0;
  memcpy(out_keys,key_a.contents,NSUInteger(count)*sizeof(uint64_t)); memcpy(out_payload,pay_a.contents,NSUInteger(count)*sizeof(uint32_t)); return 1;
 }
}
}

// Deterministic-v3 worklist foundation.  This file is isolated from the
// frozen oracle and v2.  It materializes only the dynamically delivered CSR
// edges, preserving CPU delivery rank in the low 32 bits of `key`.
#include <metal_stdlib>
using namespace metal;

struct V3Control { uint n; uint slot; uint source_count; uint edge_count; uint capacity; };

// The delayed-source sequence is already in CPU insertion order.  A scalar
// prefix is intentionally used only for K source rows (K is typically tens),
// never for the active edges.  It establishes a collision-free global order
// before the massively parallel expansion pass.
kernel void v3_source_prefix(device const long *ptr [[buffer(0)]],
                             device const int *events [[buffer(1)]],
                             device const int *event_counts [[buffer(2)]],
                             device uint *edge_offsets [[buffer(3)]],
                             device V3Control *control [[buffer(4)]],
                             uint tid [[thread_position_in_grid]]) {
  if(tid) return;
  const uint k=event_counts[control->slot];
  uint total=0;
  for(uint rank=0;rank<k;rank++) {
    edge_offsets[rank]=total;
    const int source=events[control->slot*control->n+rank];
    const ulong degree=ulong(ptr[source+1]-ptr[source]);
    // A single tick cannot exceed total graph edges (25,582,938), safely
    // inside uint32.  Preserve explicit failure state rather than truncate.
    if(degree>0xffffffffu-total) { control->edge_count=0xffffffffu; control->source_count=k; return; }
    total+=uint(degree);
  }
  edge_offsets[k]=total; control->source_count=k; control->edge_count=total;
}

// Each active outgoing edge becomes one sortable record.  key=(target,
// global_order) is unique because global_order is the packed source-rank/CSR
// ordinal.  Binary searching the K-row offset table avoids reconstructing
// source order from IDs and keeps all graph/state data on device.
kernel void v3_expand_edges(device const long *ptr [[buffer(0)]],
                            device const int *post [[buffer(1)]],
                            device const float *weight [[buffer(2)]],
                            device const int *events [[buffer(3)]],
                            device const uint *edge_offsets [[buffer(4)]],
                            device ulong *keys [[buffer(5)]],
                            device float *values [[buffer(6)]],
                            device uint *ranks [[buffer(7)]],
                            device const V3Control *control [[buffer(8)]],
                            uint order [[thread_position_in_grid]]) {
  const uint total=control->edge_count;
  if(order>=total || total>control->capacity) return;
  uint lo=0, hi=control->source_count;
  while(lo+1<hi) { const uint mid=(lo+hi)>>1; if(edge_offsets[mid]<=order) lo=mid; else hi=mid; }
  const uint rank=lo; const int source=events[control->slot*control->n+rank];
  const ulong edge=ulong(ptr[source])+ulong(order-edge_offsets[rank]);
  const uint target=uint(post[edge]);
  keys[order]=(ulong(target)<<32)|ulong(order); values[order]=weight[edge]; ranks[order]=rank;
}

// Following an integer radix sort of keys, a target boundary is independent
// of sorting stability because the complete global order is embedded in key.
kernel void v3_mark_segments(device const ulong *keys [[buffer(0)]],
                             device uchar *heads [[buffer(1)]],
                             device const V3Control *control [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if(i>=control->edge_count) return;
  heads[i]=uchar(i==0 || uint(keys[i]>>32)!=uint(keys[i-1]>>32));
}

// Four-bit least-significant-digit radix pass.  The bridge dispatches these
// three kernels for shifts 0..60.  Payload is the original packed-edge index;
// all associated arrays remain indexed by it, avoiding payload drift.
struct RadixControl { uint count; uint blocks; uint shift; };
kernel void v3_radix_hist(device const ulong *keys [[buffer(0)]], device uint *hist [[buffer(1)]], constant RadixControl &c [[buffer(2)]], uint tid [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
  threadgroup atomic_uint bins[16]; if(tid<16) atomic_store_explicit(&bins[tid],0,memory_order_relaxed); threadgroup_barrier(mem_flags::mem_threadgroup);
  uint i=block*256+tid; if(i<c.count) atomic_fetch_add_explicit(&bins[uint((keys[i]>>c.shift)&15ul)],1,memory_order_relaxed); threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid<16) hist[block*16+tid]=atomic_load_explicit(&bins[tid],memory_order_relaxed);
}
kernel void v3_radix_prefix(device const uint *hist [[buffer(0)]], device uint *offsets [[buffer(1)]], constant RadixControl &c [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  if(tid) return; uint base[16]; for(uint d=0;d<16;d++)base[d]=0; for(uint b=0;b<c.blocks;b++)for(uint d=0;d<16;d++)base[d]+=hist[b*16+d]; uint sum=0;for(uint d=0;d<16;d++){uint n=base[d];base[d]=sum;sum+=n;} for(uint b=0;b<c.blocks;b++)for(uint d=0;d<16;d++){uint n=hist[b*16+d];offsets[b*16+d]=base[d];base[d]+=n;}
}
kernel void v3_radix_scatter(device const ulong *in_keys [[buffer(0)]], device const uint *in_payload [[buffer(1)]], device ulong *out_keys [[buffer(2)]], device uint *out_payload [[buffer(3)]], device const uint *offsets [[buffer(4)]], constant RadixControl &c [[buffer(5)]], uint tid [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
  threadgroup uint digit[256],rank[256]; uint i=block*256+tid; digit[tid]=i<c.count?uint((in_keys[i]>>c.shift)&15ul):16; threadgroup_barrier(mem_flags::mem_threadgroup);
  // Stable local ranks: deliberately simple correctness-first scan. The radix
  // key is complete, so stability is not required for correctness, but makes
  // payload movement auditable.
  uint r=0;for(uint j=0;j<tid;j++)r+=digit[j]==digit[tid];rank[tid]=r;threadgroup_barrier(mem_flags::mem_threadgroup);
  if(i<c.count){uint d=digit[tid],dst=offsets[block*16+d]+rank[tid];out_keys[dst]=in_keys[i];out_payload[dst]=in_payload[i];}
}

"""Build a non-scientific execution index from the immutable verified CSR graph."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
GRAPH=ROOT/'upstream/doomfly/outputs/doom/malecns_v1/graph.npz'
OUT=ROOT/'workspace/gpu/build/target_major.npz'
def main():
 a=np.load(GRAPH,mmap_mode='r');ptr,post=a['ptr'],a['post'];n=len(a['ids']);e=len(post)
 # Original ordinal is CSR traversal position. Stable sort by target preserves
 # original ordinal within each target, exactly the requested source order.
 ordinal=np.arange(e,dtype=np.int32);order=np.argsort(post,kind='stable')
 target=post[order];offset=np.r_[0,np.cumsum(np.bincount(target,minlength=n),dtype=np.int64)]
 np.savez(OUT,target_offsets=offset.astype(np.int64),incoming_source=np.searchsorted(ptr,order,side='right').astype(np.int32)-1,incoming_weight=a['weight'][order].astype(np.float32),incoming_original_ordinal=ordinal[order])
 print({'neurons':n,'edges':e,'path':str(OUT)})
if __name__=='__main__':main()

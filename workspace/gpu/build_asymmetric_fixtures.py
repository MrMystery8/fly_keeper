"""Create deterministic retinal-only parity fixtures from audited receptor UVs."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
GRAPH=ROOT/'upstream/doomfly/outputs/doom/malecns_v1/graph.npz'
OUT=ROOT/'workspace/gpu/fixtures/asymmetric_retinal_fixtures.npz'

def main():
    g=np.load(GRAPH);uv=g['uv'];n=len(uv);base=np.zeros(n,np.float32)
    # These are explicit synthetic luminance fields, not environment state.
    left=base.copy();left[uv[:,0]<.33]=1
    center=base.copy();center[(uv[:,0]>=.33)&(uv[:,0]<=.67)]=1
    right=base.copy();right[uv[:,0]>.67]=1
    strong_left=base.copy();strong_left[uv[:,0]<.5]=1
    strong_right=base.copy();strong_right[uv[:,0]>=.5]=1
    # Deterministic nearest-receptor horizontal reflection.  This is a fixture
    # transform only; no simulator mapping or biological graph is changed.
    reflected=np.argmin((uv[:,None,0]-(1-uv[None,:,0]))**2+(uv[:,None,1]-uv[None,:,1])**2,axis=1)
    names=np.asarray(['blank','left_ball','center_ball','right_ball','strong_left','strong_right',
                      'mirrored_left','mirrored_right','mirrored_strong_left','mirrored_strong_right'])
    values=[base,left,center,right,strong_left,strong_right,left[reflected],right[reflected],strong_left[reflected],strong_right[reflected]]
    np.savez(OUT,names=names,luminance=np.stack(values),retina_indices=g['retina'],dt_ms=np.float32(20))
    print(OUT)
if __name__=='__main__':main()

"""Record only the exact R1-R6 samples supplied by a deterministic FlyKeeper frame sequence."""
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'workspace'));sys.path.insert(0,str(ROOT/'upstream/doomfly'))
from adapters.brain import MaleCNSBrain
from doom.game import retinal_samples
from experiments.flykeeper.environment import FlyKeeper
from experiments.flykeeper.renderer import render
def main():
 b=MaleCNSBrain();e=FlyKeeper(7);s=e.reset(700);items=[]
 while not s.done:items.append(retinal_samples(render(s,e),b._brain.uv));s=e.step('STAY')
 out=ROOT/'workspace/gpu/fixtures/flykeeper_retinal_sequence.npz';np.savez(out,luminance=np.asarray(items,dtype=np.float32),retina_indices=b._brain.retina,dt_ms=np.float32(20.));print(out)
if __name__=='__main__':main()

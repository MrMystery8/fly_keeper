"""CPU reference queue diagnostic: source event at t=0 delivers at t=18."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'upstream/doomfly'))
from doom.native import _f

def advance(state,steps):
 n=3;clock=np.asarray([state['clock']],np.int64);a=[state[k] for k in ('ptr','post','weight','v','g','refractory','drive','previous_drive','queue','queue_count')]+[clock]
 _f(n,*[x.ctypes.data for x in a],steps,.1,*[state[k].ctypes.data for k in ('counts','active','flags','nactive','last')]);state['clock']=int(clock[0])
def make():
 return {'ptr':np.array([0,2,3,3],np.int64),'post':np.array([1,2,2],np.int32),'weight':np.ones(3,np.float32),'v':np.array([-44.,-52.,-52.],np.float32),'g':np.zeros(3,np.float32),'refractory':np.zeros(3,np.int16),'drive':np.zeros(3,np.float32),'previous_drive':np.zeros(3,np.float32),'queue':np.zeros((19,3),np.int32),'queue_count':np.zeros(19,np.int32),'counts':np.zeros(3,np.int32),'active':np.array([0,0,0],np.int32),'flags':np.array([1,0,0],np.uint8),'nactive':np.array([1],np.int32),'last':np.full(3,-1,np.int64),'clock':0}
def test_delayed_source_event_delivery():
 s=make();advance(s,1);assert s['queue_count'][18]==1 and s['queue'][18,0]==0;assert s['refractory'][0]==22
 advance(s,17);assert s['g'][1]==0 and s['g'][2]==0
 advance(s,1);assert s['g'][1]>0 and s['g'][2]>0

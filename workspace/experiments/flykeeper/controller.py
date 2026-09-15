class MaleCNSGoalkeeper:
    """Engineering decoder: DNp20 L/R spike counts only; no game state enters it."""
    LEFT=(10162,); RIGHT=(10059,)
    def decode(self, activity):
        left=sum(activity[i]['spikes'] for i in self.LEFT);right=sum(activity[i]['spikes'] for i in self.RIGHT)
        return ('LEFT' if left>right and left else 'RIGHT' if right>left and right else 'STAY'),{'left':left,'stay':0,'right':right}
class RandomGoalkeeper:
    def __init__(self,seed):import numpy as np;self.rng=np.random.default_rng(seed)
    def decode(self,*_):
        a=str(self.rng.choice(['LEFT','STAY','RIGHT']));return a,{'left':int(a=='LEFT'),'stay':int(a=='STAY'),'right':int(a=='RIGHT')}

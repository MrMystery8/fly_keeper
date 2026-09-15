from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    ball_x: float; ball_y: float; keeper_x: float; done: bool=False; result: str|None=None

class FlyKeeper:
    """Deterministic 2D task. Only rendered pixels are supplied to the brain."""
    width=160; height=96; goal_y=84; keeper_y=78
    def __init__(self, seed=7): self.rng=np.random.default_rng(seed); self.state=None
    def reset(self, seed=None, shot_x=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        self.state=State(float(self.rng.integers(20,141) if shot_x is None else shot_x),8.,80.);return self.state
    def step(self, action):
        s=self.state; s.keeper_x=float(np.clip(s.keeper_x+{'LEFT':-6,'STAY':0,'RIGHT':6}[action],12,148));s.ball_y+=5
        if s.ball_y>=self.keeper_y:
            s.done=True;s.result='SAVE' if abs(s.ball_x-s.keeper_x)<=13 else 'GOAL'
        return s

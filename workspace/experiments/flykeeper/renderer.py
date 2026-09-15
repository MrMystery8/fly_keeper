import numpy as np
def render(state, env):
    """Rendered RGB only: neutral pitch, white goal, cyan keeper, bright ball."""
    a=np.full((env.height,env.width,3),12,dtype=np.uint8);a[env.goal_y:env.goal_y+2,10:150]=180;a[10:env.goal_y,10:12]=180;a[10:env.goal_y,148:150]=180
    x=int(state.keeper_x);a[env.keeper_y:env.keeper_y+6,x-12:x+13]=[40,210,230]
    x,y=int(state.ball_x),int(state.ball_y);a[max(0,y-3):y+4,max(0,x-3):x+4]=[255,245,220];return a

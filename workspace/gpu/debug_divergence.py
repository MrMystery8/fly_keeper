"""First-disagreement reporter for CPU vs future Metal state snapshots."""
import numpy as np
def first_divergence(cpu,gpu,ids,atol=1e-6):
    for field in ('v','g','refractory','spike'):
        a,b=np.asarray(cpu[field]),np.asarray(gpu[field]);bad=np.flatnonzero(a!=b if field in ('refractory','spike') else ~np.isclose(a,b,atol=atol,rtol=0))
        if len(bad):
            i=int(bad[0]);return {'field':field,'index':i,'neuron_id':int(ids[i]),'cpu':a[i].item(),'gpu':b[i].item(),'max_abs_error':float(np.max(np.abs(a.astype(float)-b.astype(float))))}
    return None

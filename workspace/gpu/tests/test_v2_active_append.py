"""Ordering contract for v2's parallel-update / serial-append propagation."""
import numpy as np

def phase_b(active, flags, row_targets, awakened):
    out=list(active);flags=np.asarray(flags,dtype=np.uint8).copy()
    for target,wake in zip(row_targets,awakened):
        if wake and not flags[target]: flags[target]=1;out.append(target)
    return out,flags

def test_csr_order_not_lane_or_target_order():
    active,flags=phase_b([],np.zeros(701,np.uint8),[91,3,700,12],[1,1,1,1])
    assert active==[91,3,700,12]

def test_mixed_and_previously_active_targets_do_not_duplicate():
    flags=np.zeros(701,np.uint8);flags[3]=1
    active,flags=phase_b([3],flags,[91,3,700,12,2],[1,1,1,1,0])
    assert active==[3,91,700,12]

def test_later_source_observes_prior_source_membership():
    active,flags=phase_b([],np.zeros(20,np.uint8),[9],[1])
    active,flags=phase_b(active,flags,[9,4],[1,1])
    assert active==[9,4]

def test_ordered_ring_wraps_after_18_steps():
 ring=[[] for _ in range(19)]
 delivered=[]
 for t in range(57):
  future=(t+18)%19;ring[future].extend([91,2] if t in (0,19,38) else [])
  if t in (18,37,56):
   assert ring[t%19]==[91,2]
   delivered.append(ring[t%19].copy())
  ring[t%19]=[]
 assert delivered==[[91,2],[91,2],[91,2]]

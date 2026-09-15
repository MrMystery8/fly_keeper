"""Audit CSR rows without changing the verified graph."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
 a=np.load(ROOT/'upstream/doomfly/outputs/doom/malecns_v1/graph.npz',mmap_mode='r');ptr,post=a['ptr'],a['post'];rows=[];pairs=0;maxdup=1
 for src in range(len(ptr)-1):
  row=post[ptr[src]:ptr[src+1]]
  if len(row)>1:
   _,counts=np.unique(row,return_counts=True);m=int(counts.max())
   if m>1:
    duplicate_targets=np.unique(row)[counts>1];pairs+=int((counts>1).sum());maxdup=max(maxdup,m)
    rows.append({'source_index':src,'source_id':int(a['ids'][src]),'duplicate_target_indices':[int(x) for x in duplicate_targets[:8]],'maximum_multiplicity':m})
 report={'source_rows':len(ptr)-1,'rows_with_duplicate_targets':len(rows),'duplicate_source_target_pairs':pairs,'maximum_multiplicity':maxdup,'representative_rows':rows[:10]}
 out=ROOT/'workspace/gpu/SOURCE_ROW_DUPLICATES.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

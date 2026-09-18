"""Independent, interpretable binocular Arcade Bridge V3.

Two causal linear heads deliberately use separate feature identities, temporal
windows, normalisation, and ridge penalties.  Both still inject only real DNs;
there is no direct visual-to-body path.
"""
from __future__ import annotations
import json
import numpy as np


class V3Head:
    def __init__(self, body_ids, n_windows, W, b, mu, sd):
        self.body_ids=[int(x) for x in body_ids]; self.n_windows=int(n_windows)
        self.W=np.asarray(W,float).ravel(); self.b=float(b); self.mu=np.asarray(mu,float); self.sd=np.where(np.asarray(sd,float)<1e-6,1.,np.asarray(sd,float))

    def predict(self, history, union_index):
        pos=[union_index[i] for i in self.body_ids]
        x=np.asarray(history[-self.n_windows:][::-1])[:,pos].reshape(-1)
        return float(np.tanh(((x-self.mu)/self.sd) @ self.W + self.b))


class ArcadeBinocularV3Bridge:
    """Feature history -> separate linear heads -> existing bounded DN basis."""
    def __init__(self, lateral, vertical, basis, lat_gain=3.5, vert_gain=4.5, smoothing=.2):
        self.lateral=lateral; self.vertical=vertical; self.basis=basis
        self.body_ids=list(dict.fromkeys(lateral.body_ids+vertical.body_ids)); self._index={b:i for i,b in enumerate(self.body_ids)}
        self.n_windows=max(lateral.n_windows,vertical.n_windows); self.lat_gain=float(lat_gain); self.vert_gain=float(vert_gain); self.smoothing=float(smoothing); self.enabled=True; self.reset()

    def reset(self):
        self._buf=[np.zeros(len(self.body_ids),np.float32) for _ in range(self.n_windows)]; self._raw=np.zeros(2); self._ema=np.zeros(2)

    def observe(self, brain):
        rd=brain.read(self.body_ids); self._buf.append(np.array([rd[i]["spikes"] for i in self.body_ids],np.float32)); self._buf.pop(0)

    def command(self):
        self._raw[:]=[self.lateral.predict(self._buf,self._index),self.vertical.predict(self._buf,self._index)]
        self._ema=self.smoothing*self._ema+(1-self.smoothing)*self._raw
        return float(self._ema[0]),float(max(0.,self._ema[1]))

    def inject(self, brain):
        return self.basis.inject(brain,float(self._ema[0])*self.lat_gain,float(max(0.,self._ema[1]))*self.vert_gain)


def save(path, lateral, vertical, metadata):
    np.savez(path, lat_ids=np.asarray(lateral.body_ids,np.int64),lat_windows=np.array([lateral.n_windows]),lat_W=lateral.W,lat_b=np.array([lateral.b]),lat_mu=lateral.mu,lat_sd=lateral.sd,
             vert_ids=np.asarray(vertical.body_ids,np.int64),vert_windows=np.array([vertical.n_windows]),vert_W=vertical.W,vert_b=np.array([vertical.b]),vert_mu=vertical.mu,vert_sd=vertical.sd,
             metadata=np.array([json.dumps(metadata)]))


def load(path, basis):
    d=np.load(path,allow_pickle=False)
    lat=V3Head(d["lat_ids"],int(d["lat_windows"][0]),d["lat_W"],d["lat_b"][0],d["lat_mu"],d["lat_sd"])
    vert=V3Head(d["vert_ids"],int(d["vert_windows"][0]),d["vert_W"],d["vert_b"][0],d["vert_mu"],d["vert_sd"])
    meta=json.loads(str(d["metadata"][0])); bridge=ArcadeBinocularV3Bridge(lat,vert,basis,meta["lat_gain"],meta["vert_gain"],meta["smoothing"]); return bridge,meta

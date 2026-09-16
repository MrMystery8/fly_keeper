"""Lateral-vision audit: isolate WHY the arcade lateral bridge signal is weak.

Answers the six diagnostic questions before any retrain:
  1. Do both eyes / retinal populations receive distinct useful signal?
  2. Are both hemispheres represented in the 207 bridge features?
  3. Is feature ordering / normalization identical train vs runtime?
  4. Is the negative lateral bias baked in OFFLINE or introduced at runtime?
  5. Does Metal itself reduce raw lateral decodability (apples-to-apples)?
  6. Are the teacher lateral labels intrinsically noisy / sign-reversing?

Analysis only -- does NOT modify the deployed model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from doom.game import retinal_samples
from embodiment.vision_bridge import VisionBridge
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.arcade_runtime import DECISION_MS, DECISION_S

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
MAN = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")


# =============================================================== Q1: eyes
def audit_eyes():
    """Are eye_left and eye_right cameras genuinely different views, and does the
    single-camera VisionBridge therefore MISS one hemisphere of input?"""
    import mujoco
    w = ArcadeGoalkeeperWorld(seed=90000)
    brain_uv = None
    from adapters.brain import MaleCNSBrain
    b = MaleCNSBrain(backend="cpu")
    uv = np.asarray(b._brain.uv)
    retina_ids = [int(b._brain.ids[i]) for i in b._brain.retina]
    n_ret = len(retina_ids)

    # render both eye cameras for a LEFT and a RIGHT shot and compare luminance
    out = {"n_retina_receptors": n_ret,
           "uv_x_range": [float(uv[:, 0].min()), float(uv[:, 0].max())],
           "uv_y_range": [float(uv[:, 1].min()), float(uv[:, 1].max())]}
    r = mujoco.Renderer(w.model, height=96, width=160)
    shots = {}
    for group in ("left", "right"):
        w.reset(w.sample_shot(group, 0.0))
        # step ball to mid-approach
        for _ in range(20):
            w.fly.set_command(0, 0, 0); w.step(DECISION_S)
        views = {}
        for cam in ("eye_left", "eye_right"):
            r.update_scene(w.data, camera=cam)
            img = r.render()
            lum = retinal_samples(img, uv)
            views[cam] = dict(img_mean=round(float(img.mean()), 2),
                              lum_mean=round(float(lum.mean()), 4),
                              lum_std=round(float(lum.std()), 4))
        # difference between the two eye images
        r.update_scene(w.data, camera="eye_left"); il = r.render().astype(float)
        r.update_scene(w.data, camera="eye_right"); ir = r.render().astype(float)
        views["left_vs_right_img_absdiff_mean"] = round(float(np.abs(il - ir).mean()), 2)
        shots[group] = views
    out["per_shot"] = shots
    # KEY: does the deployed VisionBridge use one camera only?
    vb = VisionBridge(w.fly, b, camera="eye_left")
    out["deployed_camera"] = vb._camera
    out["deployed_uses_single_camera"] = True
    b.close()
    return out


# ===================================================== Q2/Q3: neuron audit
def audit_neurons():
    """Hemisphere + direction-preference balance of the 207 selected neurons,
    and confirm feature identity/order matches between manifest and metadata."""
    import pandas as pd
    ann = pd.read_feather(ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather")
    id2side = dict(zip(ann["bodyId"].astype(int), ann["somaSide"].astype(str)))
    id2type = dict(zip(ann["bodyId"].astype(int), ann["type"].astype(str)))

    body = MAN["body_ids"][:207].astype(int)
    gi = MAN["graph_index"][:207].astype(int)
    dir_pref = MAN["dir_pref"][:207].astype(int)
    eff = MAN["effect_size"][:207].astype(float)

    # lateral weights from the arcade 2-output model (column 0 = lateral)
    from experiments.arcade_demo.arcade_bridge import LinearBridge2D
    m = LinearBridge2D.load(OUT / "arcade_bridge_model.npz")
    W = m.W.reshape(-1, 2)                     # (828, 2)
    n_windows = int(m.loaded_metadata.get("n_windows", 4))
    # per-neuron lateral weight = sum over its 4 windows (newest_first blocks)
    Wlat = W[:, 0].reshape(n_windows, 207).sum(axis=0)  # (207,)

    sides = [id2side.get(int(bid), "?") for bid in body]
    n_L = sum(s == "L" for s in sides); n_R = sum(s == "R" for s in sides)
    n_leftpref = int((dir_pref < 0).sum()); n_rightpref = int((dir_pref > 0).sum())
    pos_w = int((Wlat > 0).sum()); neg_w = int((Wlat < 0).sum())

    # feature-order / identity checks
    meta = m.loaded_metadata
    sel_graph = np.asarray(meta.get("sel_graph", []), dtype=int)
    order_ok = bool(sel_graph.size and np.array_equal(sel_graph[:207], gi))

    return dict(
        n_neurons=207, n_windows=n_windows,
        hemisphere=dict(L=n_L, R=n_R),
        direction_pref=dict(left_pref=n_leftpref, right_pref=n_rightpref),
        lateral_weights=dict(positive=pos_w, negative=neg_w,
                             sum=round(float(Wlat.sum()), 4),
                             mean=round(float(Wlat.mean()), 5)),
        # correlation between a neuron's CPU dir-pref and its learned lat weight
        corr_dirpref_vs_weight=round(float(np.corrcoef(dir_pref, Wlat)[0, 1]), 4),
        feature_order_matches_metadata=order_ok,
        bias_lateral=round(float(m.b[0]), 5),
        bias_vertical=round(float(m.b[1]), 5),
    )


# ===================================== Q4: offline per-class + inference parity
def audit_offline():
    """Per-class predicted lateral means on the OFFLINE dataset (held-out by
    seed), and the learned lateral bias. Determines if the negative baseline is
    baked into the trained model rather than introduced at runtime."""
    from experiments.arcade_demo.arcade_bridge import LinearBridge2D
    data = np.load(OUT / "arcade_dataset.npz", allow_pickle=False)
    X = data["features"].astype(np.float64)
    groups = np.array([g.decode() if isinstance(g, bytes) else str(g)
                       for g in data["groups"]])
    u_lat = data["u_lat"].astype(np.float64)
    m = LinearBridge2D.load(OUT / "arcade_bridge_model.npz")
    pred = np.tanh((X - m.mu) / m.sd @ m.W + m.b)   # (N,2)
    pl = pred[:, 0]
    res = {}
    for g in ("left", "center", "right"):
        mask = groups == g
        res[g] = dict(
            teacher_u_lat_mean=round(float(u_lat[mask].mean()), 4),
            pred_u_lat_mean=round(float(pl[mask].mean()), 4),
            n=int(mask.sum()))
    res["overall_pred_lat_mean"] = round(float(pl.mean()), 4)
    res["lateral_bias_b"] = round(float(m.b[0]), 5)
    # baseline-subtracted (relative to center)
    c = res["center"]["pred_u_lat_mean"]
    res["relative_to_center"] = {g: round(res[g]["pred_u_lat_mean"] - c, 4)
                                 for g in ("left", "center", "right")}
    return res


# ===================================== Q5: CPU vs Metal apples-to-apples probe
def audit_cpu_vs_metal(per_group=6, base_seed=95000, n_windows=4):
    """Collect IDENTICAL shots' features on CPU and on Metal, train identical
    diagnostic lateral probes on each, compare L/R decodability. Analysis only.
    """
    from adapters.brain import MaleCNSBrain
    body = [int(x) for x in MAN["body_ids"][:207]]

    def collect(backend):
        brain = MaleCNSBrain(backend=backend)
        Xs, ys, gs = [], [], []
        seed = base_seed
        for group in ("left", "center", "right"):
            for _ in range(per_group):
                w = ArcadeGoalkeeperWorld(seed=seed)
                vision = VisionBridge(w.fly, brain, camera="eye_left",
                                      condition="normal")
                shot = w.sample_shot(group, 0.0)
                w.reset(shot); brain.reset()
                buf = [np.zeros(len(body), np.float32) for _ in range(n_windows)]
                n = 0
                while w.result is None and n < 75:
                    vision.perceive(); brain.step(DECISION_MS)
                    rd = brain.read(body)
                    counts = np.array([rd[i]["spikes"] for i in body], np.float32)
                    buf.append(counts); buf.pop(0)
                    Xs.append(np.concatenate(buf[::-1]).astype(np.float32))
                    (ul, _), _ = teacher_command(w)
                    ys.append(ul); gs.append(group)
                    w.fly.set_command(0, 0, 0); w.step(DECISION_S); n += 1
                seed += 1
        brain.close()
        return np.asarray(Xs), np.asarray(ys), np.asarray(gs)

    def probe(X, y, g):
        # simple ridge L/R probe (left vs right frames only), report dir accuracy
        mask = g != "center"
        Xm, ym = X[mask], y[mask]
        mu, sd = Xm.mean(0), Xm.std(0); sd = np.where(sd < 1e-6, 1, sd)
        Xn = (Xm - mu) / sd
        A = Xn.T @ Xn + 300 * np.eye(Xn.shape[1])
        w_ = np.linalg.solve(A, Xn.T @ ym)
        pred = Xn @ w_
        corr = float(np.corrcoef(pred, ym)[0, 1]) if pred.std() > 1e-9 else 0.0
        dir_acc = float(np.mean(np.sign(pred) == np.sign(ym)))
        # per-class predicted means on full set
        Xn_all = (X - mu) / sd; pred_all = Xn_all @ w_
        means = {c: round(float(pred_all[g == c].mean()), 4)
                 for c in ("left", "center", "right")}
        return dict(corr=round(corr, 4), dir_acc=round(dir_acc, 4),
                    class_means=means, n=int(mask.sum()))

    Xc, yc, gc = collect("cpu")
    Xm, ym, gm = collect("metal")
    # feature-level agreement between backends on matched frames
    k = min(len(Xc), len(Xm))
    feat_corr = float(np.corrcoef(Xc[:k].ravel(), Xm[:k].ravel())[0, 1])
    return dict(cpu=probe(Xc, yc, gc), metal=probe(Xm, ym, gm),
                feature_pearson_cpu_vs_metal=round(feat_corr, 4),
                note="identical shots/labels/neurons/windows; only backend differs")


# ===================================== Q6: teacher label quality
def audit_teacher(per_group=5, base_seed=96000):
    """Sign reversals per episode + teacher target vs time-to-cross / fly pos."""
    reversals, lens = [], []
    detail = {"left": [], "right": []}
    seed = base_seed
    for group in ("left", "right"):
        for _ in range(per_group):
            w = ArcadeGoalkeeperWorld(seed=seed); shot = w.sample_shot(group, 0.0)
            w.reset(shot)
            us = []
            n = 0
            while w.result is None and n < 75:
                (ul, _), _ = teacher_command(w)
                us.append(ul)
                w.fly.set_command(0, 0, 0); w.step(DECISION_S); n += 1
            us = np.array(us)
            sign = np.sign(us[np.abs(us) > 0.05])
            rev = int(np.sum(sign[1:] != sign[:-1])) if len(sign) > 1 else 0
            reversals.append(rev); lens.append(len(us))
            detail[group].append(dict(n=len(us), reversals=rev,
                                      u_mean=round(float(us.mean()), 3),
                                      u_first=round(float(us[0]), 3),
                                      u_last=round(float(us[-1]), 3)))
            seed += 1
    return dict(mean_reversals_per_episode=round(float(np.mean(reversals)), 2),
                mean_episode_len=round(float(np.mean(lens)), 1),
                detail=detail)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="all",
                   help="all|eyes|neurons|offline|cpu_metal|teacher")
    a = p.parse_args()
    report = {}
    if a.only in ("all", "eyes"):
        print("[Q1] auditing eyes..."); report["eyes"] = audit_eyes()
    if a.only in ("all", "neurons"):
        print("[Q2/3] auditing neurons..."); report["neurons"] = audit_neurons()
    if a.only in ("all", "offline"):
        print("[Q4] auditing offline..."); report["offline"] = audit_offline()
    if a.only in ("all", "cpu_metal"):
        print("[Q5] CPU vs Metal probe (slow)..."); report["cpu_vs_metal"] = audit_cpu_vs_metal()
    if a.only in ("all", "teacher"):
        print("[Q6] auditing teacher..."); report["teacher"] = audit_teacher()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lateral_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

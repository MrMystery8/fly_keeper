"""Controller-interface audit for the frozen fullframe binocular lateral bridge.

This is deliberately an experiment harness, not a new controller.  It leaves
camera geometry, fullframe retinas, stream heads, and the V3 vertical head
untouched.  The only optional intervention is a scalar monotonic mapping from
the already-computed binocular lateral output to its existing DN input.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "workspace", ROOT / "upstream" / "doomfly"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from embodiment.vision_bridge import VisionBridge
from embodiment.mujoco_world import GOAL_LINE_X
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import Arcade2AxisDecoder, DECISION_MS, DECISION_S, MAX_DECISIONS
from experiments.arcade_demo.arcade_bridge import ArcadeLearnedBridge, LinearBridge2D, MetalSafeFeatureExtractor
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.binocular_lateral_eval import _cells, eval_oracle

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
GROUPS = ("left", "center", "right")
HEIGHTS = ("low", "mid", "high")


def make_v2():
    man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
    body = [int(x) for x in man["body_ids"][:207]]
    model = LinearBridge2D.load(OUT / "arcade_bridge_model_v2.npz")
    meta = getattr(model, "loaded_metadata", {})
    return ArcadeLearnedBridge(MetalSafeFeatureExtractor(body, int(meta.get("n_windows", 4))), model,
                               ArcadeDNBasis(), lat_gain=meta.get("lat_gain", 3.5),
                               vert_gain=meta.get("vert_gain", 4.5), enabled=True,
                               cmd_smoothing=meta.get("cmd_smoothing", .2))


def make_hybrid(name):
    return load_hybrid(name, ArcadeDNBasis())[0]


class InterfaceBridge:
    """Preserve a source bridge but expose/log its lateral DN interface.

    source supplies both raw lateral and frozen vertical signals.  ``gain`` and
    ``bias`` affect only lateral injection and are selected on development data.
    """
    def __init__(self, source, gain=1.0, bias=0.0, basis=None):
        self.source, self.gain, self.bias = source, float(gain), float(bias)
        self.basis = basis or ArcadeDNBasis()
        self.enabled = True
        self.lat_gain = source.lat_gain
        self.last_raw = self.last_cal = self.last_vert = 0.0
        self.prev_cal = self.prev_vert = 0.0
        self.last_pL = self.last_pR = 0.0

    def reset(self):
        self.source.reset()
        self.last_raw = self.last_cal = self.last_vert = 0.0
        self.prev_cal = self.prev_vert = 0.0
        self.last_pL = self.last_pR = 0.0

    def observe(self, brain):
        self.source.observe(brain)

    def command(self):
        raw, vert = self.source.command()
        self.last_raw = float(raw)
        self.last_vert = float(vert)
        self.last_cal = float(np.clip(self.gain * raw + self.bias, -1., 1.))
        self.last_pL = float(getattr(self.source, "last_pL", np.nan))
        self.last_pR = float(getattr(self.source, "last_pR", np.nan))
        self.prev_cal, self.prev_vert = self.last_cal, self.last_vert
        return self.last_cal, self.last_vert

    def inject(self, brain):
        # At controller tick t this is command t-1.  The trace labels it as
        # injected_u, not current raw output, to avoid a timing misattribution.
        return self.basis.inject(brain, self.prev_cal * self.lat_gain,
                                 self.prev_vert * self.source.vert_gain)


def ball_time(world):
    x = float(world.data.qpos[world._ball_qadr])
    vx = float(world.data.qvel[world._ball_dofadr])
    return max(0., (GOAL_LINE_X - x) / vx) if vx < -1e-8 else 0.


def run_trace(brain, seed, group, hname, hf, source_kind, hybrid_name, gain=1., bias=0.):
    world = ArcadeGoalkeeperWorld(seed=seed)
    source = make_v2() if source_kind == "v2" else make_hybrid(hybrid_name)
    bridge = InterfaceBridge(source, gain, bias)
    vision = (VisionBridge(world.fly, brain, camera="eye_left", condition="normal")
              if source_kind == "v2" else BinocularVisionBridge(world.fly, brain, condition="both", retina_map="fullframe"))
    decoder = Arcade2AxisDecoder()
    shot = world.sample_shot(group, height_frac=hf)
    world.reset(shot); brain.reset(); bridge.reset(); decoder.reset()
    rows, n = [], 0
    y0 = float(world.fly.position[1])
    while world.result is None and n < MAX_DECISIONS:
        luminance, _ = vision.perceive()
        injected_u = bridge.prev_cal
        injected_drive = injected_u * bridge.lat_gain
        bridge.inject(brain)
        step_info = brain.step(DECISION_MS)
        bridge.observe(brain)
        activity = brain.read(decoder.readout_ids())
        command, diag = decoder.decode(activity)
        bridge.command()  # causal output to be injected at next tick
        world.fly.set_command(command["forward"], command.get("turn", 0.), command["gait_on"],
                              lateral=command.get("lateral", 0.), vertical=command.get("vertical", 0.))
        world.step(DECISION_S)
        fy = float(world.fly.position[1]); vel = float(world.fly.velocity[1])
        rows.append(dict(step=n, since_launch=round(n * DECISION_S, 4), until_cross=round(ball_time(world), 4),
            raw_u=round(bridge.last_raw, 6), calibrated_u=round(bridge.last_cal, 6),
            left_stream=round(bridge.last_pL, 6), right_stream=round(bridge.last_pR, 6),
            injected_u=round(injected_u, 6), injected_drive_mv=round(injected_drive, 6),
            dn_left=round(diag["left_spikes"], 4), dn_right=round(diag["right_spikes"], 4),
            dn_diff=round(diag["right_spikes"]-diag["left_spikes"], 4),
            motor=round(command["lateral"], 6), fly_y=round(fy, 5), fly_vy=round(vel, 5),
            fly_ay=None, ball_y=round(float(world.data.qpos[world._ball_qadr+1]), 5),
            ball_ttc=round(ball_time(world), 4), retinal_mean=round(float(luminance.mean()), 6),
            spikes=int(step_info["spikes"])))
        n += 1
    diagnostic = world.outcome_diagnostics()
    return dict(seed=seed, group=group, height=hname, source=source_kind, gain=gain, bias=bias,
                result=world.result or "GOAL", keeper_contact=bool(world.keeper_contact),
                deflected=bool(world.deflected), touch_but_goal=bool(diagnostic.get("touch_but_goal", False)),
                final_fly_displacement=round(float(world.fly.position[1])-y0, 5), steps=rows)


def direction(group): return {"left": -1, "right": 1}.get(group, 0)

def latency(trace):
    desired = direction(trace["group"])
    if not desired: return dict(bridge_first=None, bridge_sustained=None, dn=None, motor=None, velocity=None)
    rows = trace["steps"]
    def first(pred):
        for r in rows:
            if pred(r): return r["since_launch"]
        return None
    # output sign uses the bridge convention; motor sign is opposite it.
    correct_u = lambda r: desired * r["raw_u"] > .03
    sustained = lambda i: all(correct_u(r) for r in rows[i:min(len(rows), i+3)])
    return dict(bridge_first=first(correct_u), bridge_sustained=next((r["since_launch"] for i,r in enumerate(rows) if sustained(i)), None),
                dn=first(lambda r: desired * r["dn_diff"] > .1),
                motor=first(lambda r: desired * r["motor"] < -.03),
                velocity=first(lambda r: desired * r["fly_vy"] < -.02))


def summary(traces):
    cells = defaultdict(list)
    for t in traces: cells[(t["group"], t["height"])].append(t)
    def rate(ts, pred): return round(sum(pred(t) for t in ts)/max(1,len(ts)), 3)
    out = dict(n=len(traces), saves=sum(t["result"]=="SAVE" for t in traces),
               overall=rate(traces, lambda t:t["result"]=="SAVE"),
               matrix_3x3={f"{g}/{h}": f"{sum(t['result']=='SAVE' for t in cells[g,h])}/{len(cells[g,h])}" for g in GROUPS for h in HEIGHTS},
               by_group={g:rate([t for t in traces if t["group"]==g], lambda t:t["result"]=="SAVE") for g in GROUPS},
               by_height={h:rate([t for t in traces if t["height"]==h], lambda t:t["result"]=="SAVE") for h in HEIGHTS},
               keeper_contact=rate(traces, lambda t:t["keeper_contact"]), touch_but_goal=rate(traces, lambda t:t["touch_but_goal"]),
               deflected_save=rate(traces, lambda t:t["deflected"] and t["result"]=="SAVE"),
               mean_displacement=round(float(np.mean([t["final_fly_displacement"] for t in traces])),4))
    return out


def distribution(traces, field="raw_u"):
    ans={}
    for g in GROUPS:
      for h in HEIGHTS:
        x=np.array([r[field] for t in traces if t["group"]==g and t["height"]==h for r in t["steps"]], float)
        if len(x): ans[f"{g}/{h}"]=dict(mean=round(float(x.mean()),4), std=round(float(x.std()),4), min=round(float(x.min()),4), max=round(float(x.max()),4), abs_mean=round(float(abs(x).mean()),4), near_zero=round(float((abs(x)<.03).mean()),3), saturation=round(float((abs(x)>=.98).mean()),3), sign_accuracy=round(float(np.mean(np.sign(x)==direction(g))),3) if direction(g) else None)
    return ans


def aggregate_latency(traces):
    ans={}
    for key, ts in [(g,[t for t in traces if t["group"]==g]) for g in GROUPS] + [(h,[t for t in traces if t["height"]==h]) for h in HEIGHTS]:
        vals=[latency(t) for t in ts]
        ans[key]={k:round(float(np.nanmean([np.nan if v[k] is None else v[k] for v in vals])),4) for k in vals[0]}
    return ans


def transfer(traces):
    xs=np.array([r["injected_u"] for t in traces for r in t["steps"]]); ys=np.array([r["motor"] for t in traces for r in t["steps"]]); ds=np.array([r["dn_diff"] for t in traces for r in t["steps"]])
    return dict(injected_to_dn_slope=round(float(np.polyfit(xs,ds,1)[0]),4), injected_to_motor_slope=round(float(np.polyfit(xs,ys,1)[0]),4), injected_saturation=round(float((abs(xs)>=.98).mean()),3), dn_zero=round(float((abs(ds)<.1).mean()),3))


def failures(traces):
    misses=[t for t in traces if t["result"]!="SAVE"]; counts=defaultdict(int)
    for t in misses:
        d=direction(t["group"]); rows=t["steps"]
        if not d: counts["F_center_or_interception"]+=1; continue
        correct=[r for r in rows if d*r["raw_u"]>.03]
        if not correct: counts["A_wrong_direction"]+=1
        elif latency(t)["velocity"] is None: counts["C_correct_but_too_late"]+=1
        elif max(abs(r["motor"]) for r in rows)<.15: counts["B_correct_but_weak"]+=1
        elif any(d*r["raw_u"]<-.03 for r in rows[len(rows)//2:]): counts["D_stops_or_reverses"]+=1
        else: counts["F_reaches_region_or_interception"]+=1
    return {k:round(v/max(1,len(misses)),3) for k,v in sorted(counts.items())}


def evaluate(brain, cells, source, hybrid, gain=1., bias=0.):
    # MuJoCo model/data allocations are native; prompt collection keeps a long
    # 90-shot matrix bounded rather than accumulating one arena per episode.
    ans=[]
    for c in cells:
        ans.append(run_trace(brain,*c,source,hybrid,gain,bias))
        gc.collect()
    return ans


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--per-cell",type=int,default=10); ap.add_argument("--base-seed",type=int,default=90000); ap.add_argument("--hybrid",default="arcade_binocular_lateral_hybrid.npz"); ap.add_argument("--dev-per-cell",type=int,default=5); ap.add_argument("--fixed-gain",type=float,default=None); ap.add_argument("--fixed-bias",type=float,default=None); a=ap.parse_args()
    # A stratified held-out set of 90 and a separately seeded, same-size dev set.
    test=_cells(a.per_cell,a.base_seed); dev=_cells(a.dev_per_cell,a.base_seed+100000)
    from adapters.brain import MaleCNSBrain
    brain=MaleCNSBrain(backend="metal"); t0=time.perf_counter()
    fixed = a.fixed_gain is not None or a.fixed_bias is not None
    if fixed and (a.fixed_gain is None or a.fixed_bias is None):
        raise ValueError("supply both --fixed-gain and --fixed-bias")
    v2_dev = [] if fixed else evaluate(brain,dev,"v2",a.hybrid)
    bi_dev = [] if fixed else evaluate(brain,dev,"binocular",a.hybrid)
    # Scalar-only dev calibration.  It matches the useful v2 output scale, not
    # save outcomes: therefore no simulation state, shot class, or test episode
    # participates in fitting.  Bias makes dev CENTER neutral before matching
    # the left/right absolute command magnitude.
    selected_from_development = not fixed
    if fixed:
        gain, bias = float(a.fixed_gain), float(a.fixed_bias)
    else:
        dev_v2 = np.array([r["raw_u"] for t in v2_dev if t["group"] != "center" for r in t["steps"]])
        dev_bi_lr = np.array([r["raw_u"] for t in bi_dev if t["group"] != "center" for r in t["steps"]])
        dev_bi_center = np.array([r["raw_u"] for t in bi_dev if t["group"] == "center" for r in t["steps"]])
        bias = float(np.clip(-dev_bi_center.mean(), -.15, .15))
        gain = float(np.clip(np.mean(abs(dev_v2)) / max(1e-6, np.mean(abs(dev_bi_lr + bias))), .5, 4.0))
    # A final evaluation may supply the already frozen development choice. It
    # is never refit from these test cells.
    candidates=[("frozen" if not selected_from_development else "moment-match",gain,bias)]
    v2=evaluate(brain,test,"v2",a.hybrid); bi=evaluate(brain,test,"binocular",a.hybrid); cal=evaluate(brain,test,"binocular",a.hybrid,gain,bias)
    # Swaps deliberately use identical source and interface implementations: both
    # production paths reduce to ArcadeDNBasis -> fixed DN decoder.  These runs
    # establish whether any hidden downstream path differs.
    v2_hybrid=evaluate(brain,test,"v2",a.hybrid); bi_v2=evaluate(brain,test,"binocular",a.hybrid)
    result=dict(protocol=dict(dev_n=len(dev), test_n=len(test), matched_reset_corrected=True, frozen=["camera geometry/FOV","fullframe retina_map","separate streams + late fusion","vertical pathway"], calibration=dict(form="clip(a*u+b)", a=float(gain),b=float(bias),selection=("pre-frozen development value; no test fit" if not selected_from_development else "stratified development only"),candidates=len(candidates))), oracle=eval_oracle(test), development=dict(v2=summary(v2_dev),binocular=summary(bi_dev)), heldout=dict(v2_one_eye=summary(v2), binocular_uncalibrated=summary(bi), binocular_calibrated=summary(cal), v2_through_hybrid_downstream=summary(v2_hybrid), binocular_through_v2_downstream=summary(bi_v2)), signals=dict(v2=dict(raw=distribution(v2),calibrated=distribution(v2,"calibrated_u"),latency=aggregate_latency(v2),transfer=transfer(v2)), binocular=dict(raw=distribution(bi),calibrated=distribution(cal,"calibrated_u"),latency=aggregate_latency(bi),transfer=transfer(bi)), center_neutrality=dict(raw={g:round(float(np.mean([r['raw_u'] for t in bi if t['group']==g for r in t['steps']])),4) for g in GROUPS}, calibrated={g:round(float(np.mean([r['calibrated_u'] for t in cal if t['group']==g for r in t['steps']])),4) for g in GROUPS})), failures=dict(v2=failures(v2),binocular=failures(bi),calibrated=failures(cal)), traces=dict(v2=v2,binocular=bi,calibrated=cal), wall_seconds=round(time.perf_counter()-t0,1))
    brain.close(); (OUT/"binocular_controller_interface_audit.json").write_text(json.dumps(result,indent=2)); print(json.dumps({"calibration":result["protocol"]["calibration"],"heldout":result["heldout"]},indent=2))

if __name__=="__main__": main()

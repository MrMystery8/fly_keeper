"""Frozen model-artifact integrity + version checks for the interactive demo.

The interactive game must load the EXACT frozen learned bridge validated in the
scientific experiment (tag ``learned-visual-dn-bridge-complete``). This module
verifies, before the game builds a Learned Bridge controller, that:

  1. the on-disk artifacts (bridge_model.npz, visual_manifest.npz) are
     bit-for-bit the frozen ones (SHA-256 checksums),
  2. the MaleCNS connectome graph is the expected fixed graph (SHA-256),
  3. the frozen metadata inside bridge_model.npz still describes the validated
     model (207 neurons, 4 x 20 ms windows, 829 params, gain 3.5, smoothing 0.8,
     drive 25 mV, newest_first feature order),

and it assembles a single, versioned ``ModelArtifact`` object that records
every field the runtime needs (feature/body IDs, ordering, temporal windows,
normalization, weights, bias, output transform, gain, DN IDs + opponent basis,
current bounds, and model/version metadata).

The expected checksums below are the frozen values recorded at the
``learned-visual-dn-bridge-complete`` state. If an artifact legitimately changes
(e.g. a re-freeze under a NEW scientific tag), re-run ``python -m
workspace.experiments.interactive_demo.artifact_integrity --print-checksums`` and
update EXPECTED_CHECKSUMS deliberately -- this is a tripwire, not a silent auto.

Nothing here retrains or tunes the model; it only reads, hashes, and validates.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

OUT = ROOT / "workspace" / "outputs" / "learned_bridge"
GRAPH_PATH = ROOT / "upstream" / "doomfly" / "outputs" / "doom" / "malecns_v1" / "graph.npz"

# --- interactive-demo model version tag -----------------------------------
# Ties this game build to the exact frozen scientific state it was validated
# against. Bump ONLY when intentionally adopting a newly re-frozen model.
MODEL_VERSION = "learned-bridge-v1"
SCIENTIFIC_TAG = "learned-visual-dn-bridge-complete"

# --- frozen artifact checksums (SHA-256), captured at the frozen tag -------
EXPECTED_CHECKSUMS = {
    "bridge_model.npz": "7cd477b9ed597d78db04c4286a34b74454fcfb3a9ff3c839d49093df1f97a9a0",
    "visual_manifest.npz": "de67c07edcdb0841ebfc1c435cca435781f8520eca16b5349224f6203ce689ee",
    # The MaleCNS connectome graph (native fixed weights). The adapter also
    # checks this SHA-256 independently at brain construction.
    "graph.npz": "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b",
}

# --- frozen metadata invariants (the validated 829-parameter bridge) -------
EXPECTED_META = {
    "n_windows": 4,
    "n_neurons": 207,
    "n_params": 829,
    "target_offset": 4,
    "best_alpha": 300,
    "runtime_gain": 3.5,
    "cmd_smoothing": 0.8,
    "drive_mv": 25.0,
}
EXPECTED_FEATURE_ORDER = "newest_first"
# Real MaleCNS descending-neuron IDs used as the frozen opponent motor basis.
EXPECTED_LEFT_DN = (10162, 10527)
EXPECTED_RIGHT_DN = (10059, 555871)


class IntegrityError(RuntimeError):
    """Raised when a frozen artifact fails checksum or metadata validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ModelArtifact:
    """Everything the runtime needs to rebuild the frozen bridge, with provenance.

    This is a reproducible, self-describing record: it carries the frozen
    weights/normalization/ordering AND the metadata + checksums that prove it is
    the validated model. The game reconstructs the bridge ONLY from this object,
    never from ad hoc defaults.
    """

    model_version: str
    scientific_tag: str
    # frozen linear model
    w: np.ndarray
    b: float
    mu: np.ndarray
    sd: np.ndarray
    feature_order: str
    output_transform: str
    # frozen feature selection / temporal windowing
    n_neurons: int
    n_windows: int
    body_ids: np.ndarray          # selected optic-lobe body IDs (order matters)
    graph_index: np.ndarray       # graph positions for the selected neurons
    dir_pref: np.ndarray
    effect_size: np.ndarray
    # frozen runtime scalars
    runtime_gain: float
    cmd_smoothing: float
    # frozen DN motor basis
    left_dn: tuple
    right_dn: tuple
    drive_mv: float
    max_abs_mv: float
    # provenance
    n_params: int
    checksums: dict
    metadata: dict = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        return int(self.n_neurons * self.n_windows)

    def summary(self) -> dict:
        return {
            "model_version": self.model_version,
            "scientific_tag": self.scientific_tag,
            "n_neurons": self.n_neurons,
            "n_windows": self.n_windows,
            "feature_dim": self.feature_dim,
            "n_params": self.n_params,
            "output_transform": self.output_transform,
            "feature_order": self.feature_order,
            "runtime_gain": self.runtime_gain,
            "cmd_smoothing": self.cmd_smoothing,
            "left_dn": list(self.left_dn),
            "right_dn": list(self.right_dn),
            "drive_mv": self.drive_mv,
            "max_abs_mv": self.max_abs_mv,
            "checksums": self.checksums,
        }


def verify_checksums(*, strict: bool = True) -> dict:
    """Return {name: {expected, actual, ok}} for each frozen artifact.

    strict=True raises IntegrityError on any mismatch/missing file.
    """
    files = {
        "bridge_model.npz": OUT / "bridge_model.npz",
        "visual_manifest.npz": OUT / "visual_manifest.npz",
        "graph.npz": GRAPH_PATH,
    }
    report = {}
    problems = []
    for name, path in files.items():
        expected = EXPECTED_CHECKSUMS.get(name)
        if not path.exists():
            report[name] = {"expected": expected, "actual": None, "ok": False,
                            "error": "missing file"}
            problems.append(f"{name}: missing at {path}")
            continue
        actual = sha256_file(path)
        ok = (actual == expected)
        report[name] = {"expected": expected, "actual": actual, "ok": ok}
        if not ok:
            problems.append(f"{name}: checksum mismatch "
                            f"(expected {expected}, got {actual})")
    if strict and problems:
        raise IntegrityError(
            "Frozen artifact integrity check FAILED:\n  - "
            + "\n  - ".join(problems)
            + "\n\nThe interactive demo must load the EXACT frozen bridge validated "
              f"at tag '{SCIENTIFIC_TAG}'. Refusing to continue.")
    return report


def _check_meta(meta: dict) -> list:
    problems = []
    for key, expected in EXPECTED_META.items():
        actual = meta.get(key)
        if actual is None:
            problems.append(f"metadata missing '{key}' (expected {expected})")
        elif isinstance(expected, float):
            if not np.isclose(float(actual), expected, atol=1e-9):
                problems.append(f"metadata '{key}'={actual} != expected {expected}")
        elif int(actual) != int(expected):
            problems.append(f"metadata '{key}'={actual} != expected {expected}")
    return problems


def load_verified_artifact(*, strict: bool = True) -> ModelArtifact:
    """Verify checksums + metadata and assemble the frozen ModelArtifact.

    This is the ONLY path the game uses to obtain the frozen model description.
    """
    from experiments.learned_bridge.bridge import LinearBridgeModel

    checks = verify_checksums(strict=strict)

    model = LinearBridgeModel.load(OUT / "bridge_model.npz")
    meta = dict(getattr(model, "loaded_metadata", {}) or {})
    man = np.load(OUT / "visual_manifest.npz")

    problems = _check_meta(meta)
    if str(model.feature_order) != EXPECTED_FEATURE_ORDER:
        problems.append(f"feature_order '{model.feature_order}' != "
                        f"'{EXPECTED_FEATURE_ORDER}'")
    n_params_actual = int(model.w.size + 1)
    if n_params_actual != EXPECTED_META["n_params"]:
        problems.append(f"n_params {n_params_actual} != {EXPECTED_META['n_params']}")

    n_neurons = int(meta.get("n_neurons", 207))
    n_windows = int(meta.get("n_windows", 4))
    if model.w.size != n_neurons * n_windows:
        problems.append(f"weight dim {model.w.size} != n_neurons*n_windows "
                        f"({n_neurons}*{n_windows})")

    # The frozen selection order must match the manifest the runtime slices.
    body_ids = man["body_ids"][:n_neurons].astype(np.int64)
    graph_index = man["graph_index"][:n_neurons].astype(np.int64)
    sel_body = np.asarray(meta.get("sel_body", []), dtype=np.int64)
    if sel_body.size and not np.array_equal(sel_body[:n_neurons], body_ids):
        problems.append("visual_manifest body order != frozen metadata sel_body "
                        "order (neuron ordering changed)")

    if strict and problems:
        raise IntegrityError(
            "Frozen bridge metadata validation FAILED:\n  - "
            + "\n  - ".join(problems)
            + f"\n\nExpected the validated model from tag '{SCIENTIFIC_TAG}'.")

    return ModelArtifact(
        model_version=MODEL_VERSION,
        scientific_tag=SCIENTIFIC_TAG,
        w=np.asarray(model.w, dtype=np.float64),
        b=float(model.b),
        mu=np.asarray(model.mu, dtype=np.float64),
        sd=np.asarray(model.sd, dtype=np.float64),
        feature_order=str(model.feature_order),
        output_transform="tanh",
        n_neurons=n_neurons,
        n_windows=n_windows,
        body_ids=body_ids,
        graph_index=graph_index,
        dir_pref=man["dir_pref"][:n_neurons].astype(np.int64),
        effect_size=man["effect_size"][:n_neurons].astype(np.float64),
        runtime_gain=float(meta.get("runtime_gain", 3.5)),
        cmd_smoothing=float(meta.get("cmd_smoothing", 0.8)),
        left_dn=EXPECTED_LEFT_DN,
        right_dn=EXPECTED_RIGHT_DN,
        drive_mv=float(meta.get("drive_mv", 25.0)),
        max_abs_mv=30.0,
        n_params=n_params_actual,
        checksums={k: v["actual"] for k, v in checks.items()},
        metadata=meta,
    )


def main():
    import argparse

    p = argparse.ArgumentParser(description="Verify frozen bridge artifacts.")
    p.add_argument("--print-checksums", action="store_true",
                   help="print current on-disk checksums (for re-freezing)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    a = p.parse_args()

    if a.print_checksums:
        files = {"bridge_model.npz": OUT / "bridge_model.npz",
                 "visual_manifest.npz": OUT / "visual_manifest.npz",
                 "graph.npz": GRAPH_PATH}
        current = {n: (sha256_file(pth) if pth.exists() else None)
                   for n, pth in files.items()}
        print(json.dumps(current, indent=2))
        return

    try:
        art = load_verified_artifact(strict=True)
    except IntegrityError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
    out = art.summary()
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"[PASS] frozen bridge verified ({art.model_version}, "
              f"tag {art.scientific_tag})")
        print(f"  neurons={art.n_neurons} windows={art.n_windows} "
              f"feature_dim={art.feature_dim} params={art.n_params}")
        print(f"  gain={art.runtime_gain} smoothing={art.cmd_smoothing} "
              f"drive={art.drive_mv} mV")
        print(f"  transform={art.output_transform} order={art.feature_order}")
        print(f"  left_dn={art.left_dn} right_dn={art.right_dn}")
        for name, cs in art.checksums.items():
            print(f"  {name}: {cs[:16]}... OK")


if __name__ == "__main__":
    main()

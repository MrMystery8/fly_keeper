# MaleCNS v1.0 reusable baseline — setup report

Prepared 2026-09-14 as a local, fixed-weight baseline. This is a simulation using measured MaleCNS connectivity, not a conscious fly, an exact digital fly, or a validated reproduction of fly physiology or behavior.

## Environment

- Machine: MacBook Air, Apple M4, 24 GB unified memory
- Architecture: `arm64`
- macOS: 26.6.2 (build 25G83)
- Xcode Command Line Tools: `/Library/Developer/CommandLineTools`
- Compiler: Apple Clang 21.0.0, arm64 target
- Git: 2.55.0; Homebrew: 6.0.15
- System Python: 3.14.6; dedicated neural environment: Python 3.11.15 at `upstream/doomfly/.venv-neural`

The Python 3.11 environment uses upstream pins: Brian2 2.5.1, NumPy 1.24.4, Pandas 2.0.3, SciPy 1.10.1, PyArrow 20.0.0, Numba 0.61.2, ViZDoom 1.3.0, Pytest 8.4.2, Cython 0.29.37, SymPy 1.12.1, Joblib 1.5.2, and Setuptools 80.9.0. `pip check` passed. Brian2 emits upstream deprecation warnings concerning `pkg_resources` and Pyparsing; these are warnings, not test failures, and pins were not changed.

## Repository

- Checkout: `upstream/doomfly/`
- Source remote: `https://github.com/nftechie/doomfly.git`
- Cloned source commit: `71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33`
- Local working branch: `generic-malecns-base`

No upstream simulator source files were edited. Generated import/audit/build artifacts are present in the checkout as expected from the official preparation pipeline.

## MaleCNS data

Dataset: `male-cns:v1.0`, MaleCNS v1.0, brain and ventral nerve cord. Files were downloaded exclusively from the URLs in `doom/datasets.json`, stored in `upstream/doomfly/connectome_data/malecns_v1/`, and checked against `data-provenance/malecns_v1/source.lock.json` before import.

| File | Bytes | SHA-256 | Result |
| --- | ---: | --- | --- |
| annotations.feather | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` | verified |
| neurotransmitters.feather | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` | verified |
| edges.feather | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` | verified |

Sources are the three exact Google-hosted MaleCNS v1.0 URLs in `doom/datasets.json` under the official Janelia dataset page: <https://male-cns.janelia.org/download/>.

Import/audit result: **166,700 retained neurons, 25,582,938 retained directed edges, and 124,177,617 retained synaptic contacts**. It retains 10,299,701 weight-one edges and 101 self-edges. The importer examined 151,856,684 released rows; non-retained endpoints are accounted for by the upstream annotated-neuron policy, not arbitrary pruning. `doom.audit_data` passed every raw/normalized/runtime edge and provenance check.

## Native build

`python -m doom.build_kernel` completed with Apple Clang. Result: `outputs/doom/libneural.dylib`, a Mach-O 64-bit arm64 dynamic library.

- Kernel source SHA-256: `2dc0939d5efb5b3b53551a7836b24385e30976b6dfb8ece9edbb0a6983bed337`
- Binary SHA-256: `38966e272d607f3bfcc35006e56d7f8c8718e5281212abd2072c8a397201e950`
- Model revision: `lif-r2-refractory-write-protection`
- Compile flags: `-O3 -std=c++17 -shared -fPIC`

The runtime verifies these hashes before loading the library.

## Tests and runtime

Ran:

```sh
.venv-neural/bin/python -m pytest tests/test_doom.py tests/test_doom_reference.py tests/test_connectome.py tests/test_doom_live_training.py -q
```

Result: **27 passed, 0 failed, 0 skipped** in 5.05 seconds. The generic real-graph smoke test also passed: it initialized the 166,700-neuron graph, stimulated a genuine mapped R1-R6 biological neuron ID, advanced 10 milliseconds, observed a spike from that real neuron, and exited cleanly.

Fixed baseline server was run without `--learning`, `--model experimental-v6`, or reward input. It bound only to `127.0.0.1:8766`; both `/health` and `/state` returned `status: running`. State confirmed the intact graph, `dt_ms: 0.1`, `reward.mode: off`, and `plasticity: false`. It shut down cleanly with Ctrl-C.

Short M4 benchmark (single full-graph baseline process):

- Neural speed: 29.2857 simulated seconds in 56.017 wall seconds, **0.523× real time** (~5,230 0.1-ms integration steps/s)
- Last native brain step: 46.012 ms for roughly 28.5 ms neural time
- Server process RSS: 702,256 KiB (~686 MiB); CPU: ~916% across cores
- System-reported swap at measurement: 428.56 MiB used. This is whole-system swap, so it cannot be attributed solely to DoomFly.
- Disk: raw + normalized dataset 1.6 GB; prepared graph 239 MB; Python environment 786 MB

This is a short practicality measurement, not an endurance or scientific-performance claim.

## Architecture map

- Connectome loading: `doom/connectome.py` (`import_graph`, `normalize_nodes`, `index_edges`) reads annotations, neurotransmitters, and edges, validates source IDs, and writes normalized Arrow/Feather data. `doom/datasets.json` is the locked source registry.
- Graph preparation: `doom/prepare.py` (`prepare`) generates the complete CSR graph at `outputs/doom/malecns_v1/graph.npz`, derives the mapped R1-R6 retinal projection, and writes `manifest.json`.
- Neural runtime: `doom/engine.py` (`Brain`, `advance`) defines the documented LIF state/update and 0.1-ms timestep. `doom/native.py` (`NativeBrain`) binds the compiled all-edge kernel. `doom/kernel.cpp` is the kernel source; `doom/build_kernel.py` builds and hash-binds it.
- Sensory input: `doom/game.py` (`retinal_samples`) samples game RGB. `doom/server.py` supplies those values to `NativeBrain.step`; the R1-R6 mapping is derived in `doom/prepare.py` from contacts onto annotated L1/L2/L3. This visual mapping is an inferred engineering proxy, not a measured retina model.
- Output decoding: `doom/engine.py` (`NeuralControls.decode`) converts readout spike counts into Doom turn/movement/fire. The BCI path uses DNp20 and DNpe017; its assignments/gains are engineered controller choices, not validated natural motor semantics.
- Reward/plasticity: baseline reward stimulus is in `doom/reward.py`; optional experimental v6 learning is isolated in `doom_learning_v6/` and invoked from `doom/server.py` only with `--model experimental-v6 --learning`. It remains OFF. It is explicitly unvalidated and has not established learned Doom behavior.
- Checkpoints: fixed baseline integrity-checked recovery is in `doom/checkpoint.py`; experimental-training checkpointing is separate in `doom/training_checkpoint.py`.

## Generic adapter

The only new integration layer is `workspace/adapters/`. `brain.py` wraps the unmodified `doom.native.NativeBrain`, accepts genuine mapped retinal biological IDs, steps the real graph, and reads activity. `sensory.py` and `output.py` are documented placeholders; no game, robot, RL policy, or arbitrary decoder was added. `workspace/config/default.yaml` keeps `plasticity.enabled: false` and leaves populations empty.

## Limitations

- Wiring, synapse counts, annotations, and transmitter predictions are biological data; LIF dynamics, signs for uncertain neurotransmitters, thresholds, delays, sensory encoding, reward, plasticity, and decoding are modeling assumptions.
- The adapter’s initial `stimulate` intentionally supports only currently mapped retinal IDs. A reviewed direct-current mechanism and generic checkpoint schema are future integration work; neither was smuggled into the scientific core.
- No learning was enabled or claimed. The upstream experimental-v6 candidate is not evidence of validated fly learning.
- Free disk space after setup was about 7.2 GB; avoid simultaneous full-graph jobs and keep room for checkpoints/audits.

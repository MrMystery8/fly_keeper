# Frozen serial Metal reference identity

## METAL_REFERENCE_STATUS = FROZEN_PARITY_ORACLE

## DETERMINISTIC_V2_AUTHORIZED = true

This identifies the **test-only** `metal-reference-serial` correctness oracle.
It is not a selectable FlyKeeper backend and must change only with a documented
bug demonstrated by CPU-versus-Metal parity tests. Authorization applies only
to the tested fixed-weight workloads recorded in `fixtures/reference_manifest.json`.

| Artifact | SHA-256 |
|---|---|
| `metal/MaleCNS.metal` | `634a0e422a1de18ca7e76deadfe0dcd9e79848d891b12a13826740f8174527ca` |
| `metal/bridge.mm` | `2f0503ec2e3a1a1708d3a99e02461eaf27a962442a2e4f5e6c0f24cf0f67f74c` |
| `metal_reference.py` | `21d8ae1cebb32aea6ccb8f90a62f6a4dcc61ddec483dfbef48ea2ef568d0b8e5` |
| `run_fixture_parity.py` | `36988dc4f4c97137b3e1c1180eec152fff53d4e8b4b0806652f0877422caf7c2` |
| `flykeeper_parity.py` | `37af0989279da1679309bbacdb61b0f24d1182363241af60a5d36060f8902aa1` |
| `build.py` | `852f65fc5aff58db6f40ace6531f3949f3e82c0c8d70961d1d72b548d1672bf2` |
| `build/MaleCNS.metallib` | `5cce10a7af1fab51f8d1a506109f8b6da40c1382d66e7fc9eb9092055ed52af4` |
| `build/libmalecns_metal_bridge.dylib` | `c5fb5956040981da5b4145aa1fa8151cd9156dcd8ecfcac3c8742397df7d63bc` |

The corresponding immutable CPU source identity is recorded in
`REFERENCE_CPU_HASHES.md`.

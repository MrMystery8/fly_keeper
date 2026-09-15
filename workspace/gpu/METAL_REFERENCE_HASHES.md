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
| Historical `build/libmalecns_metal_bridge.dylib` | `c5fb5956040981da5b4145aa1fa8151cd9156dcd8ecfcac3c8742397df7d63bc` |

## Current-toolchain bridge identity

`HISTORICAL_SERIAL_BRIDGE_SHA256` remains
`c5fb5956040981da5b4145aa1fa8151cd9156dcd8ecfcac3c8742397df7d63bc`.
It must not be replaced or erased.

Under the current Apple toolchain, unchanged frozen bridge source reproducibly
builds to `CURRENT_TOOLCHAIN_SERIAL_BRIDGE_SHA256 =
f3bc6ca88852029bb2fcdc89fc3e972a9a98cee8c215be1e75b943683e675ba4`.
This is a linker/SDK binary identity difference, not a source or
simulation-semantics change. It is accepted only together with frozen
behavioral parity verification.

The exact build command is the frozen `build.py` invocation:
`xcrun clang++ -std=c++17 -fobjc-arc -dynamiclib metal/bridge.mm -framework
Foundation -framework Metal -o build/libmalecns_metal_bridge.dylib`, with
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.

Current environment recorded 2026-09-15: macOS 26.6.2 (25G83), Darwin
25.6.0 arm64; Apple clang 21.0.0 (clang-2100.1.1.101); SDK 26.5;
ld-1267. The current dylib is arm64 Mach-O, links Foundation and Metal, has
LC_BUILD_VERSION minOS 26.0, and UUID `AC3ADE66-F889-3DF9-B92F-EDA75EEC7EBC`.
The historical bytes are unavailable, so a byte-level comparison is not
possible. Frozen CPU/source/metallib/fixture identities remain authoritative.

The corresponding immutable CPU source identity is recorded in
`REFERENCE_CPU_HASHES.md`.

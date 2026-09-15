"""Build non-oracle deterministic Metal backends without touching the frozen build path."""
import os
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    out = root / "build"
    out.mkdir(exist_ok=True)
    env = {**os.environ, "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer"}
    for name in ("MaleCNS_v2", "MaleCNS_v3"):
        source = root / "metal" / f"{name}.metal"
        air = out / f"{name}.air"
        metallib = out / f"{name}.metallib"
        subprocess.run(["xcrun", "metal", "-fno-fast-math", "-c", str(source), "-o", str(air)], check=True, env=env)
        subprocess.run(["xcrun", "metallib", str(air), "-o", str(metallib)], check=True, env=env)
    subprocess.run([
        "xcrun", "clang++", "-std=c++17", "-fobjc-arc", "-dynamiclib",
        str(root / "metal" / "bridge_v2.mm"), "-framework", "Foundation", "-framework", "Metal",
        "-o", str(out / "libmalecns_metal_v2.dylib"),
    ], check=True, env=env)


if __name__ == "__main__":
    main()

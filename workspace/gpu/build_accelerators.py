"""Build non-oracle deterministic Metal backends without touching the frozen build path."""
import os
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    out = root / "build"
    out.mkdir(exist_ok=True)
    env = {**os.environ, "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer"}
    for name in ("MaleCNS_v2", "MaleCNS_v3", "MaleCNS_v4", "MaleCNS_v5", "MaleCNS_v6"):
        source = root / "metal" / f"{name}.metal"
        air = out / f"{name}.air"
        metallib = out / f"{name}.metallib"
        math_flag = "-ffast-math" if name == "MaleCNS_v5" else "-fno-fast-math"
        subprocess.run(["xcrun", "metal", math_flag, "-c", str(source), "-o", str(air)], check=True, env=env)
        subprocess.run(["xcrun", "metallib", str(air), "-o", str(metallib)], check=True, env=env)
    subprocess.run([
        "xcrun", "clang++", "-std=c++17", "-fobjc-arc", "-dynamiclib",
        str(root / "metal" / "bridge_v2.mm"), "-framework", "Foundation", "-framework", "Metal",
        "-o", str(out / "libmalecns_metal_v2.dylib"),
    ], check=True, env=env)
    subprocess.run([
        "xcrun", "clang++", "-std=c++17", "-fobjc-arc", "-dynamiclib",
        str(root / "metal" / "bridge_v3.mm"), "-framework", "Foundation", "-framework", "Metal",
        "-o", str(out / "libmalecns_metal_v3.dylib"),
    ], check=True, env=env)
    subprocess.run([
        "xcrun", "clang++", "-std=c++17", "-fobjc-arc", "-dynamiclib",
        str(root / "metal" / "bridge_v4.mm"), "-framework", "Foundation", "-framework", "Metal",
        "-o", str(out / "libmalecns_metal_v4.dylib"),
    ], check=True, env=env)
    subprocess.run([
        "xcrun", "clang++", "-std=c++17", "-fobjc-arc", "-dynamiclib",
        str(root / "metal" / "bridge_v5.mm"), "-framework", "Foundation", "-framework", "Metal",
        "-o", str(out / "libmalecns_metal_v5.dylib"),
    ], check=True, env=env)
    subprocess.run([
        "xcrun", "clang++", "-std=c++17", "-fobjc-arc", "-dynamiclib",
        str(root / "metal" / "bridge_v6.mm"), "-framework", "Foundation", "-framework", "Metal",
        "-o", str(out / "libmalecns_metal_v6.dylib"),
    ], check=True, env=env)


if __name__ == "__main__":
    main()

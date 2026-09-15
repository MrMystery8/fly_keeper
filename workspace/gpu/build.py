"""Metal toolchain preflight; never builds or alters the CPU reference."""
import os, subprocess
from pathlib import Path
def main():
    root=Path(__file__).resolve().parent;out=root/'build';out.mkdir(exist_ok=True)
    env={**os.environ,'DEVELOPER_DIR':'/Applications/Xcode.app/Contents/Developer'}
    try: print(subprocess.check_output(['xcrun','metal','-v'],text=True,stderr=subprocess.STDOUT,env=env))
    except subprocess.CalledProcessError as e: raise SystemExit(e.output)
    # The serial oracle compares native CPU float32 state after every tick.
    # Disable Metal fast-math contraction so it cannot silently fuse a native
    # multiply/add into a numerically different FMA.
    subprocess.run(['xcrun','metal','-fno-fast-math','-c',str(root/'metal/MaleCNS.metal'),'-o',str(out/'MaleCNS.air')],check=True,env=env)
    subprocess.run(['xcrun','metallib',str(out/'MaleCNS.air'),'-o',str(out/'MaleCNS.metallib')],check=True,env=env)
    subprocess.run(['xcrun','clang++','-std=c++17','-fobjc-arc','-dynamiclib',str(root/'metal/bridge.mm'),'-framework','Foundation','-framework','Metal','-o',str(out/'libmalecns_metal_bridge.dylib')],check=True,env=env)
    print(out/'MaleCNS.metallib')
if __name__=='__main__':main()

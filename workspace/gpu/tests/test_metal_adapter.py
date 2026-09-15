import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
subprocess.run([str(ROOT/'upstream/doomfly/.venv-neural/bin/python'),'-m','workspace.gpu.build_accelerators'],cwd=ROOT,check=True)

from workspace.adapters.brain import MaleCNSBrain

def test_optional_metal_adapter_advances_and_reads():
    brain=MaleCNSBrain(backend='metal')
    try:
        before=brain._brain.sim_ms
        result=brain.step(1.0)
        read=brain.read([10162,10059])
        assert result['backend']=='metal'
        assert brain._brain.sim_ms==before+1.0
        assert set(read)=={10162,10059}
    finally:
        brain.close()

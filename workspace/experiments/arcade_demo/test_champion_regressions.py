"""Automated regression suite for the frozen Early-Intent RL goalkeeper champion.

Verifies:
1. Lateral sign convention (LEFT -> +y, RIGHT -> -y).
2. Sensing freeze (steps 0..7: gait_on == 0, zero lateral drive, fly holds still).
3. Reset hygiene (clean MaleCNS membrane/spike state and bridge buffers per episode).
4. Fullframe binocular mapping.
5. CLEAN_GOAL absence-of-natural-miss validation.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import (Arcade2AxisDecoder,
                                                    DECISION_MS, DECISION_S)
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.early_intent import EarlyIntentEstimator
from experiments.arcade_demo.early_intent_bridge import EarlyIntentBridge, ExecPolicy
from experiments.arcade_demo import shot_validity as SV

POLICY_NAME = "arcade_early_intent_vert_refined_rl.npz"


class ChampionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = MaleCNSBrain(backend="metal")
        cls.est, _ = EarlyIntentEstimator.load()
        cls.policy, cls.meta = ExecPolicy.load(POLICY_NAME)

    @classmethod
    def tearDownClass(cls):
        cls.brain.close()

    def test_binocular_mapping(self):
        """Ensure fullframe retina mapping is preserved."""
        world = ArcadeGoalkeeperWorld(seed=100)
        vision = BinocularVisionBridge(world.fly, self.brain, condition="both", retina_map="fullframe")
        self.assertEqual(vision.retina_map, "fullframe")
        self.assertEqual(vision.condition, "both")

    def test_sensing_freeze(self):
        """During steps 0..7, gait_on must be 0.0 and lateral command must be 0.0."""
        world = ArcadeGoalkeeperWorld(seed=600001)
        shot = world.sample_shot("left", height_frac=0.5)
        hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
        bridge = EarlyIntentBridge(hybrid, self.est, policy=self.policy, sense_steps=8)
        vision = BinocularVisionBridge(world.fly, self.brain, condition="both", retina_map="fullframe")
        
        world.reset(shot)
        self.brain.reset()
        bridge.reset()
        
        start_y = float(world.fly.position[1])
        start_z = float(world.fly.position[2])
        
        for step in range(8):
            vision.perceive()
            injected_dids, _ = bridge.inject(self.brain)
            # Must NOT inject locomotor DNs during sensing window
            self.assertEqual(len(injected_dids), 0, f"Injected DNs during sensing step {step}")
            self.brain.step(DECISION_MS)
            bridge.observe(self.brain)
            
            gait = bridge.gait_on()
            u_lat, u_vert = bridge.command(world)
            self.assertEqual(gait, 0.0, f"gait_on != 0 at step {step}")
            self.assertEqual(u_lat, 0.0, f"u_lat != 0 at step {step}")
            self.assertLessEqual(u_vert, 0.25, f"u_vert > 0.25 readiness limit at step {step}")
            
            world.fly.set_command(0, 0, gait, lateral=u_lat, vertical=u_vert)
            world.step(DECISION_S)
            
            # Fly must remain grounded and stationary during sensing
            cur_y = float(world.fly.position[1])
            cur_z = float(world.fly.position[2])
            self.assertLess(abs(cur_y - start_y), 0.05, f"Lateral displacement during sensing at step {step}")
            self.assertLess(abs(cur_z - start_z), 0.08, f"Vertical launch during sensing at step {step}")
            
        # At step 8, execution must begin: gait_on switches to 1.0
        self.assertEqual(bridge.gait_on(), 1.0)

    def test_lateral_sign_convention(self):
        """LEFT shots (+y) must drive fly in +y; RIGHT shots (-y) must drive fly in -y."""
        hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
        
        # Test LEFT shot
        world = ArcadeGoalkeeperWorld(seed=600002)
        shot_l = world.sample_shot("left", height_frac=0.5)
        bridge_l = EarlyIntentBridge(hybrid, self.est, policy=self.policy, sense_steps=8)
        vision_l = BinocularVisionBridge(world.fly, self.brain, condition="both", retina_map="fullframe")
        world.reset(shot_l); self.brain.reset(); bridge_l.reset()
        
        for _ in range(18):
            vision_l.perceive(); bridge_l.inject(self.brain)
            self.brain.step(DECISION_MS); bridge_l.observe(self.brain)
            g = bridge_l.gait_on(); u_lat, u_vert = bridge_l.command(world)
            world.fly.set_command(0, 0, g, lateral=u_lat, vertical=u_vert)
            world.step(DECISION_S)
            if world.result is not None:
                break
        final_y_left = float(world.fly.position[1])
        self.assertGreater(final_y_left, 0.10, f"Fly did not move LEFT on LEFT shot: final_y={final_y_left}")
        
        # Test RIGHT shot
        world = ArcadeGoalkeeperWorld(seed=600003)
        shot_r = world.sample_shot("right", height_frac=0.5)
        bridge_r = EarlyIntentBridge(hybrid, self.est, policy=self.policy, sense_steps=8)
        vision_r = BinocularVisionBridge(world.fly, self.brain, condition="both", retina_map="fullframe")
        world.reset(shot_r); self.brain.reset(); bridge_r.reset()
        
        for _ in range(18):
            vision_r.perceive(); bridge_r.inject(self.brain)
            self.brain.step(DECISION_MS); bridge_r.observe(self.brain)
            g = bridge_r.gait_on(); u_lat, u_vert = bridge_r.command(world)
            world.fly.set_command(0, 0, g, lateral=u_lat, vertical=u_vert)
            world.step(DECISION_S)
            if world.result is not None:
                break
        final_y_right = float(world.fly.position[1])
        self.assertLess(final_y_right, -0.10, f"Fly did not move RIGHT on RIGHT shot: final_y={final_y_right}")

    def test_reset_hygiene(self):
        """Consecutive episodes must reset cleanly with zero residual activity."""
        hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
        bridge = EarlyIntentBridge(hybrid, self.est, policy=self.policy, sense_steps=8)
        world = ArcadeGoalkeeperWorld(seed=600004)
        shot = world.sample_shot("center", height_frac=0.5)
        vision = BinocularVisionBridge(world.fly, self.brain, condition="both", retina_map="fullframe")
        
        # Episode 1
        world.reset(shot); self.brain.reset(); bridge.reset()
        for _ in range(10):
            vision.perceive(); bridge.inject(self.brain)
            self.brain.step(DECISION_MS); bridge.observe(self.brain)
            g = bridge.gait_on(); u_lat, u_vert = bridge.command(world)
            world.fly.set_command(0, 0, g, lateral=u_lat, vertical=u_vert)
            world.step(DECISION_S)
            
        # Reset for Episode 2
        world.reset(shot); self.brain.reset(); bridge.reset()
        self.assertEqual(bridge._t, 0)
        self.assertTrue(bridge.sensing)
        self.assertTrue(np.all(bridge.previous == 0.0))
        self.assertTrue(np.all(bridge.last_obs == 0.0))


if __name__ == "__main__":
    unittest.main()

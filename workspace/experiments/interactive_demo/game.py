"""Fruit-Fly-Brain-Powered Goalkeeper -- interactive 3D penalty demo (pygame).

Launch:
    upstream/doomfly/.venv-neural/bin/python -m workspace.experiments.interactive_demo.game

The human takes penalties against a fly goalkeeper that is physically embodied in
MuJoCo. In the main "Learned Bridge Fly" mode the save/goal outcome is produced
by the EXACT frozen scientific chain:

    human penalty -> rendered 3D scene -> fly eye camera -> R1-R6 retina
      -> full fixed MaleCNS -> 207 frozen optic-lobe features
      -> frozen 829-parameter bridge -> real MaleCNS descending neurons
      -> existing decoder / CPG -> articulated MuJoCo body -> SAVE / GOAL

The learned controller NEVER sees ball position/velocity/aim/power. The scientific
simulation runs in a background thread at its true (slower-than-real-time) cadence;
the UI renders whatever the sim has produced (Section 18/19).

Controls:
    Mouse move (over pitch)  aim left/right (continuous)
    Up / Down  or  scroll    adjust shot power
    Space / click Shoot      take the penalty
    R                        reset / next shot
    N                        restart match
    1..5                     select goalkeeper mode
    B                        toggle bridge ON/OFF (learned mode)
    V                        cycle vision condition (normal/blind/mirror/shuffle)
    D                        toggle science / neural debug panel
    S                        run deterministic Science-Demo (fixed shot set)
    Esc / Q                  quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.interactive_demo.modes import (MODES, MODE_LABELS,
                                                MODE_DESCRIPTIONS,
                                                VISION_CONDITIONS, VISION_LABELS,
                                                PlasticityUnavailable,
                                                PLASTICITY_HISTORICAL)
from experiments.interactive_demo.session import SimSession
from experiments.interactive_demo.shots import science_demo_shots
from experiments.interactive_demo.artifact_integrity import (load_verified_artifact,
                                                             IntegrityError)

# ---- layout -------------------------------------------------------------
WIN_W, WIN_H = 1120, 880
SCENE_X, SCENE_Y, SCENE_W, SCENE_H = 20, 70, 640, 480
PANEL_X = SCENE_X + SCENE_W + 20
EYE_W, EYE_H = 240, 144

# ---- colours ------------------------------------------------------------
BG = (16, 18, 24)
PANEL = (26, 29, 38)
INK = (232, 234, 240)
MUTE = (150, 156, 170)
ACCENT = (90, 200, 230)
GOODC = (70, 210, 110)
BADC = (230, 90, 90)
WARN = (240, 190, 80)
SELBG = (44, 60, 78)

SELECTABLE_MODES = ("natural", "learned", "random", "heuristic", "plasticity")


class Button:
    def __init__(self, rect, label, key=None):
        self.rect = rect
        self.label = label
        self.key = key

    def hit(self, pos):
        return self.rect.collidepoint(pos)


class Game:
    def __init__(self, mode="learned", vision="normal", match_len=5):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("Fruit-Fly-Brain-Powered Goalkeeper")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo,Consolas,monospace", 15)
        self.font_s = pygame.font.SysFont("Menlo,Consolas,monospace", 12)
        self.font_b = pygame.font.SysFont("Menlo,Consolas,monospace", 22, bold=True)
        self.font_big = pygame.font.SysFont("Menlo,Consolas,monospace", 44, bold=True)

        print("[game] verifying frozen bridge artifacts ...")
        self.artifact = load_verified_artifact(strict=True)
        print(f"[game] OK: {self.artifact.model_version} "
              f"({self.artifact.n_params} params, tag {self.artifact.scientific_tag})")

        self.mode = mode
        self.vision = vision
        self.match_len = match_len
        self.show_debug = True
        self.science_demo_queue = []       # remaining (seed, group, shot)
        self.science_demo_active = False

        # player input state
        self.aim = 0.0        # -1..+1
        self.power = 0.6      # 0..1
        self.session = None
        self._new_session()
        self._reset_match()

    # ----------------------------------------------------------- session mgmt
    def _new_session(self):
        if self.session is not None:
            self.session.stop()
            self.session = None
        try:
            self.session = SimSession(self.mode, vision=self.vision,
                                      artifact=self.artifact)
            self.session_error = None
        except PlasticityUnavailable as exc:
            self.session = None
            self.session_error = str(exc)

    def _reset_match(self):
        self.shots = 0
        self.saves = 0
        self.goals = 0
        self.last_result = None
        self.science_demo_active = False
        self.science_demo_queue = []
        self.science_scores = {}

    def _select_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        self._new_session()
        self._reset_match()

    def _cycle_vision(self):
        i = VISION_CONDITIONS.index(self.vision)
        self.vision = VISION_CONDITIONS[(i + 1) % len(VISION_CONDITIONS)]
        self._new_session()
        self._reset_match()

    def _toggle_bridge(self):
        if self.session is not None and self.mode == "learned":
            snap = self.session.snapshot()
            self.session.set_bridge_enabled(not snap.bridge_on)

    # -------------------------------------------------------------- gameplay
    def _busy(self):
        return self.session is not None and self.session.is_busy()

    def _shoot(self):
        if self.session is None or self._busy():
            return
        if self.science_demo_active:
            return
        self.session.start_shot(self.aim, self.power)
        self._pending_result = True

    def _run_science_demo(self):
        """Queue a fixed deterministic shot set for repeatable comparison."""
        if self.session is None or self._busy():
            return
        self._reset_match()
        self.science_demo_active = True
        self.science_demo_queue = science_demo_shots(n_per_group=2, base_seed=90000)
        self._pending_result = True

    def _account_result(self, result):
        self.shots += 1
        if result == "SAVE":
            self.saves += 1
        else:
            self.goals += 1
        self.last_result = result

    def _tick_logic(self):
        """Advance match bookkeeping based on the sim snapshot."""
        if self.session is None:
            return
        snap = self.session.snapshot()
        # A shot just finished?
        if getattr(self, "_pending_result", False) and snap.phase == "done":
            self._account_result(snap.result)
            self._pending_result = False
            if self.science_demo_active:
                self.science_scores.setdefault(self.mode, []).append(snap.result)
        # science-demo: fire the next queued shot when idle/done
        if (self.science_demo_active and not self._busy()
                and not getattr(self, "_pending_result", False)):
            if self.science_demo_queue:
                seed, group, shot = self.science_demo_queue.pop(0)
                self.session.start_shot_spec(shot)
                self._pending_result = True
            else:
                self.science_demo_active = False

    # ------------------------------------------------------------- rendering
    def _surf_from_rgb(self, rgb):
        # pygame wants (W, H, 3) with swapped axes for make_surface
        return self.pg.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

    def _text(self, s, x, y, font=None, color=INK):
        font = font or self.font
        self.screen.blit(font.render(s, True, color), (x, y))

    def _panel(self, x, y, w, h, title=None):
        self.pg.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=8)
        if title:
            self._text(title, x + 12, y + 8, self.font, ACCENT)

    def draw(self):
        pg = self.pg
        self.screen.fill(BG)
        # Frames are rendered inside the sim worker (macOS single-GL-thread);
        # the UI just blits the latest snapshot.
        snap = self.session.snapshot() if self.session else None

        # --- title bar ---
        self._text("Fruit-Fly-Brain-Powered Goalkeeper", 20, 16, self.font_b, INK)
        self._text(f"{self.artifact.n_params} learned params  |  166,700-neuron "
                   "MaleCNS  |  frozen: " + self.artifact.scientific_tag,
                   20, 46, self.font_s, MUTE)

        # --- scene view ---
        self._panel(SCENE_X, SCENE_Y, SCENE_W, SCENE_H)
        if snap and snap.scene_rgb is not None:
            self.screen.blit(self._surf_from_rgb(snap.scene_rgb), (SCENE_X, SCENE_Y))
        elif self.session_error:
            self._wrap(self.session_error, SCENE_X + 20, SCENE_Y + 30,
                       SCENE_W - 40, WARN)

        # --- aim / power bars over the scene ---
        self._draw_aim_power()

        # --- result overlay ---
        if snap and snap.phase == "done" and self.last_result:
            self._draw_result_overlay(self.last_result, snap)
        elif snap and snap.phase == "running":
            pace = f"sim/real {snap.sim_realtime_ratio:.2f}x  (decision "
            pace += f"{snap.decision}/{snap.max_decisions})"
            self._text(pace, SCENE_X + 12, SCENE_Y + SCENE_H - 26, self.font_s,
                       WARN)

        # --- right column ---
        self._draw_mode_selector()
        self._draw_scoreboard()
        if self.show_debug:
            self._draw_debug_panel(snap)
        else:
            self._draw_info_panel()

        # --- fly-eye inset ---
        self._draw_fly_eye(snap)

        # --- controls hint ---
        self._text("Mouse aim  |  Up/Down power  |  Space shoot  |  R reset  |  "
                   "N match  |  1-5 mode  |  B bridge  |  V vision  |  D debug  |  "
                   "S science-demo  |  Q quit",
                   20, WIN_H - 22, self.font_s, MUTE)

        pg.display.flip()

    def _wrap(self, text, x, y, w, color=INK, font=None):
        font = font or self.font_s
        words = text.split()
        line = ""
        yy = y
        for word in words:
            test = (line + " " + word).strip()
            if font.size(test)[0] > w:
                self.screen.blit(font.render(line, True, color), (x, yy))
                yy += font.get_height() + 2
                line = word
            else:
                line = test
        if line:
            self.screen.blit(font.render(line, True, color), (x, yy))
        return yy + font.get_height()

    def _draw_aim_power(self):
        pg = self.pg
        # aim bar
        bx, by, bw = SCENE_X + 40, SCENE_Y + SCENE_H + 14, SCENE_W - 80
        pg.draw.rect(self.screen, PANEL, (bx, by, bw, 14), border_radius=7)
        cx = int(bx + (self.aim * 0.5 + 0.5) * bw)
        pg.draw.circle(self.screen, ACCENT, (cx, by + 7), 9)
        self._text("AIM  right", bx - 4, by + 18, self.font_s, MUTE)
        self._text("left", bx + bw - 30, by + 18, self.font_s, MUTE)
        # power bar
        py = by + 40
        pg.draw.rect(self.screen, PANEL, (bx, py, bw, 14), border_radius=7)
        pw = int(self.power * bw)
        pg.draw.rect(self.screen, WARN, (bx, py, pw, 14), border_radius=7)
        self._text(f"POWER {self.power*100:3.0f}%", bx - 4, py + 18, self.font_s,
                   MUTE)

    def _draw_result_overlay(self, result, snap):
        color = GOODC if result == "SAVE" else BADC
        s = self.font_big.render(result, True, color)
        self.screen.blit(s, (SCENE_X + SCENE_W // 2 - s.get_width() // 2,
                             SCENE_Y + 20))
        info = (f"aim {snap.shot_aim:+.2f}  power {snap.shot_power*100:.0f}%  "
                f"| {MODE_LABELS[self.mode]}")
        self._text(info, SCENE_X + 16, SCENE_Y + 76, self.font_s, INK)
        self._text("Press R for next shot", SCENE_X + 16, SCENE_Y + 96,
                   self.font_s, MUTE)

    def _draw_mode_selector(self):
        x, y, w = PANEL_X, SCENE_Y, WIN_W - PANEL_X - 20
        h = 168
        self._panel(x, y, w, h, "Goalkeeper Mode")
        self.mode_buttons = []
        yy = y + 30
        for i, m in enumerate(SELECTABLE_MODES):
            rect = self.pg.Rect(x + 10, yy, w - 20, 24)
            sel = (m == self.mode)
            if sel:
                self.pg.draw.rect(self.screen, SELBG, rect, border_radius=5)
            label = f"{i+1}. {MODE_LABELS[m]}"
            col = ACCENT if sel else INK
            if m == "plasticity":
                label += "  [historical]"
                col = MUTE
            if m == "heuristic":
                label += "  (oracle)"
            self._text(label, x + 16, yy + 4, self.font, col)
            self.mode_buttons.append(Button(rect, m))
            yy += 26
        # description
        self._wrap(MODE_DESCRIPTIONS[self.mode], x + 12, yy + 2, w - 24, MUTE)
        # vision condition line
        self._text(f"Vision: {VISION_LABELS[self.vision]}", x + 12, y + h - 20,
                   self.font_s, WARN if self.vision != "normal" else MUTE)

    def _draw_scoreboard(self):
        x, y, w = PANEL_X, SCENE_Y + 180, WIN_W - PANEL_X - 20
        h = 118
        self._panel(x, y, w, h, "Match")
        rate = (self.saves / self.shots * 100) if self.shots else 0.0
        lines = [
            f"Shots     {self.shots}" + (f" / {self.match_len}"
                                         if not self.science_demo_active else ""),
            f"Saved     {self.saves}",
            f"Goals     {self.goals}",
            f"Save rate {rate:5.1f}%",
        ]
        yy = y + 30
        for ln in lines:
            self._text(ln, x + 14, yy, self.font)
            yy += 20
        if not self.science_demo_active and self.match_len and self.shots >= self.match_len:
            self._text("Match complete - press N", x + 14, y + h - 18,
                       self.font_s, GOODC)

    def _draw_info_panel(self):
        x, y, w = PANEL_X, SCENE_Y + 306, WIN_W - PANEL_X - 20
        h = WIN_H - y - 40
        self._panel(x, y, w, h, "What you're seeing")
        txt = ("The fly sees the penalty through its own simulated visual "
               "system. In Learned Bridge mode: camera -> fly retina -> MaleCNS "
               "connectome -> small learned visual-to-motor bridge -> real "
               "descending neurons -> articulated fly body. The learned bridge "
               "has 829 trainable parameters; the native MaleCNS synaptic "
               "weights remain fixed. The natural fly connectome did not "
               "naturally know how to play football -- the bridge is an "
               "engineered learned transformation across an experimentally "
               "identified visual-to-motor bottleneck. Press D for the neural "
               "debug panel.")
        self._wrap(txt, x + 12, y + 32, w - 24, MUTE)

    def _draw_debug_panel(self, snap):
        x, y, w = PANEL_X, SCENE_Y + 306, WIN_W - PANEL_X - 20
        h = WIN_H - y - 40
        self._panel(x, y, w, h, "Science / Neural Debug")
        if snap is None:
            self._text("(mode unavailable)", x + 14, y + 32, self.font, WARN)
            if self.mode == "plasticity":
                self._wrap(f"Historical result {PLASTICITY_HISTORICAL['save_rate']:.0%}: "
                           + PLASTICITY_HISTORICAL["note"], x + 12, y + 56,
                           w - 24, MUTE)
            return
        yy = y + 30
        rows = [
            ("controller", MODE_LABELS[self.mode]),
            ("vision", VISION_LABELS[self.vision]),
            ("bridge", "ON" if snap.bridge_on else "OFF"),
            ("bridge u", f"{snap.bridge_u:+.3f}" if snap.is_neural else "n/a"),
            ("left DN spk", f"{snap.left_dn_spikes:.0f}" if snap.is_neural else "n/a"),
            ("right DN spk", f"{snap.right_dn_spikes:.0f}" if snap.is_neural else "n/a"),
            ("DN asym", f"{snap.dn_asymmetry:+.3f}" if snap.is_neural else "n/a"),
            ("inj current", f"{snap.inj_total_mv:.1f} mV" if snap.bridge_on else "0 mV"),
            ("retina mean", f"{snap.retinal_mean:.3f}" if snap.is_neural else "n/a"),
            ("brain spikes", f"{snap.brain_spikes}" if snap.is_neural else "n/a"),
            ("move", snap.move),
            ("decision", f"{snap.decision}/{snap.max_decisions}"),
            ("outcome", snap.result or "-"),
        ]
        for k, v in rows:
            self._text(k, x + 14, yy, self.font_s, MUTE)
            self._text(str(v), x + 130, yy, self.font_s, INK)
            yy += 17
        # bridge u history sparkline + L/R drive bars
        if snap.is_neural and snap.u_history:
            self._sparkline(snap.u_history, x + 14, yy + 6, w - 28, 40,
                            "bridge command u  (left<0 | right>0)")
            yy += 60
            self._lr_bars(snap.left_dn_spikes, snap.right_dn_spikes,
                          x + 14, yy + 6, w - 28, 28)
        if self.science_demo_active or self.science_scores:
            self._draw_science_scores(x + 14, y + h - 24)

    def _sparkline(self, series, x, y, w, h, label):
        pg = self.pg
        self._text(label, x, y - 2, self.font_s, MUTE)
        y += 14
        pg.draw.rect(self.screen, (20, 22, 30), (x, y, w, h), border_radius=4)
        pg.draw.line(self.screen, (60, 64, 78), (x, y + h // 2),
                     (x + w, y + h // 2), 1)
        if len(series) < 2:
            return
        n = len(series)
        pts = []
        for i, v in enumerate(series):
            px = x + int(i / max(1, n - 1) * w)
            py = y + int(h / 2 - np.clip(v, -1, 1) * (h / 2 - 2))
            pts.append((px, py))
        pg.draw.lines(self.screen, ACCENT, False, pts, 2)

    def _lr_bars(self, left, right, x, y, w, h):
        pg = self.pg
        total = max(1.0, left + right)
        half = w // 2 - 6
        lw = int(half * min(1.0, left / total))
        rw = int(half * min(1.0, right / total))
        mid = x + w // 2
        pg.draw.rect(self.screen, (60, 200, 220), (mid - 6 - lw, y, lw, h))
        pg.draw.rect(self.screen, (220, 80, 200), (mid + 6, y, rw, h))
        self._text("L DN", x, y + h + 2, self.font_s, MUTE)
        self._text("R DN", x + w - 34, y + h + 2, self.font_s, MUTE)

    def _draw_science_scores(self, x, y):
        parts = []
        for m, res in self.science_scores.items():
            rate = sum(r == "SAVE" for r in res) / len(res) if res else 0
            parts.append(f"{MODE_LABELS[m].split()[0]} {rate:.0%}({len(res)})")
        self._text("Science-Demo: " + "  ".join(parts), x, y, self.font_s, GOODC)

    def _draw_fly_eye(self, snap):
        x = SCENE_X
        y = SCENE_Y + SCENE_H + 96
        self._panel(x, y, EYE_W + 24, EYE_H + 40, "FLY VISION")
        if snap and snap.eye_rgb is not None:
            surf = self._surf_from_rgb(snap.eye_rgb)
            surf = self.pg.transform.scale(surf, (EYE_W, EYE_H))
            self.screen.blit(surf, (x + 12, y + 26))
        cond = VISION_LABELS.get(self.vision, self.vision)
        self._text(cond, x + 12, y + EYE_H + 28, self.font_s, MUTE)

        # retina luminance panel to the right of the eye
        rx = x + EYE_W + 40
        self._panel(rx, y, 200, EYE_H + 40, "R1-R6 RETINA")
        if snap and snap.retina_lum is not None and self.session:
            self._draw_retina(snap.retina_lum, rx + 12, y + 26, 176, EYE_H)

    def _draw_retina(self, lum, x, y, w, h):
        uv = self.session._uv if self.session else None
        if uv is None:
            return
        pg = self.pg
        # clamp the panel fully inside the window
        w = max(1, min(w, WIN_W - x - 2))
        h = max(1, min(h, WIN_H - y - 2))
        pg.draw.rect(self.screen, (10, 10, 14), (x, y, w, h))
        xs = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
        ys = np.clip((uv[:, 1] * (h - 1)).astype(int), 0, h - 1)
        vv = np.clip(np.asarray(lum) * 255, 0, 255).astype(int)
        arr = pg.surfarray.pixels3d(self.screen)
        n = min(len(xs), len(vv))
        for i in range(n):
            px, py = x + int(xs[i]), y + int(ys[i])
            if 0 <= px < WIN_W and 0 <= py < WIN_H:
                c = int(vv[i])
                arr[px, py] = (c, c, c)
        del arr

    # ---------------------------------------------------------------- events
    def _handle_mouse_aim(self, pos):
        bx, bw = SCENE_X + 40, SCENE_W - 80
        if SCENE_X <= pos[0] <= SCENE_X + SCENE_W and SCENE_Y <= pos[1] <= SCENE_Y + SCENE_H + 70:
            frac = np.clip((pos[0] - bx) / bw, 0, 1)
            self.aim = float(frac * 2 - 1)

    def handle_events(self):
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return False
            if e.type == pg.MOUSEMOTION:
                self._handle_mouse_aim(e.pos)
            if e.type == pg.MOUSEWHEEL:
                self.power = float(np.clip(self.power + e.y * 0.05, 0, 1))
            if e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                # mode buttons
                clicked_mode = False
                for b in getattr(self, "mode_buttons", []):
                    if b.hit(e.pos):
                        self._select_mode(b.key)
                        clicked_mode = True
                if not clicked_mode:
                    self._handle_mouse_aim(e.pos)
                    self._shoot()
            if e.type == pg.KEYDOWN:
                if e.key in (pg.K_ESCAPE, pg.K_q):
                    return False
                if e.key == pg.K_SPACE:
                    self._shoot()
                if e.key == pg.K_r:
                    self.last_result = None
                if e.key == pg.K_n:
                    self._reset_match()
                if e.key == pg.K_b:
                    self._toggle_bridge()
                if e.key == pg.K_v:
                    self._cycle_vision()
                if e.key == pg.K_d:
                    self.show_debug = not self.show_debug
                if e.key == pg.K_s:
                    self._run_science_demo()
                if e.key in (pg.K_UP,):
                    self.power = float(np.clip(self.power + 0.05, 0, 1))
                if e.key in (pg.K_DOWN,):
                    self.power = float(np.clip(self.power - 0.05, 0, 1))
                if pg.K_1 <= e.key <= pg.K_5:
                    self._select_mode(SELECTABLE_MODES[e.key - pg.K_1])
        return True

    def run(self, max_frames=None, auto_shot=False):
        """Main loop. max_frames/auto_shot are for the headless UI smoke test."""
        running = True
        frame = 0
        fired = False
        while running:
            running = self.handle_events()
            if auto_shot and not fired and frame == 5:
                self._shoot()
                fired = True
            self._tick_logic()
            self.draw()
            self.clock.tick(30)
            frame += 1
            if max_frames is not None and frame >= max_frames:
                running = False
        if self.session is not None:
            self.session.stop()
        self.pg.quit()


def selftest():
    """Headless smoke test: no display, exercises the session + one shot each mode."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    art = load_verified_artifact(strict=True)
    print(f"[selftest] artifact OK {art.model_version} ({art.n_params} params)")
    for mode in ("random", "heuristic", "natural", "learned"):
        sess = SimSession(mode, vision="normal", artifact=art)
        sess.start_shot(aim=-0.7, power=0.5)
        t0 = time.perf_counter()
        while sess.is_busy() and time.perf_counter() - t0 < 90:
            time.sleep(0.1)
        snap = sess.snapshot()   # frames were rendered inside the worker
        assert snap.scene_rgb is not None, f"{mode}: no scene frame rendered"
        assert snap.eye_rgb is not None, f"{mode}: no eye frame rendered"
        print(f"[selftest] {mode:10s} -> {snap.result} "
              f"(dec {snap.decision}, u={snap.bridge_u:+.3f}, "
              f"scene {snap.scene_rgb.shape}, eye {snap.eye_rgb.shape}, "
              f"sim/real {snap.sim_realtime_ratio:.2f}x)")
        sess.stop()
    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=SELECTABLE_MODES, default="learned")
    p.add_argument("--vision", choices=VISION_CONDITIONS, default="normal")
    p.add_argument("--match-len", type=int, default=5)
    p.add_argument("--selftest", action="store_true",
                   help="headless session smoke test (no window)")
    p.add_argument("--ui-selftest", action="store_true",
                   help="headless full-UI smoke test (dummy video, auto-shoots)")
    a = p.parse_args()

    if a.selftest:
        selftest()
        return
    if a.ui_selftest:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        g = Game(mode=a.mode, vision=a.vision, match_len=a.match_len)
        g.run(max_frames=400, auto_shot=True)
        print(f"[ui-selftest] PASS  shots={g.shots} saves={g.saves} "
              f"goals={g.goals} last={g.last_result}")
        return

    try:
        Game(mode=a.mode, vision=a.vision, match_len=a.match_len).run()
    except IntegrityError as exc:
        print(f"[FATAL] frozen artifact check failed:\n{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

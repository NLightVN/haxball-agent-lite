"""
play.py — Interactive Play
Watch or play against the AI in a pure Python environment.

Controls (while window is open):
    SPACE   — pause / resume
    R       — restart episode immediately
    Q / ESC — quit

Human opponent controls (OPPONENT="Human"):
    Arrow keys      — move
    Enter / RCtrl   — kick
"""

import math
import sys
import time
import os

import numpy as np

# Đảm bảo working directory luôn là thư mục chứa script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Configuration ─────────────────────────────────────────────────────────────
MODE = "A2v2"                           # "A1" | "A2v1" | "A2v2"
PLAY_AS = "T1"                          # For A2v1/A2v2: 'T1', 'T2'

# A1 configs
A1_MODEL_PATH      = "models/a1_fix_checkpoints/snapshot_2000000.zip"  # Agent chính (A1)
A1_OPP_MODEL_PATH  = "models/a1_checkpoints/snapshot_6000000.zip"      # Opponent model (A1)

# A2v1/A2v2 configs
A2_T1_MODEL_PATH   = r"models\a2_t1_checkpoints\snapshot_1000000.zip"
A2_T2_MODEL_PATH   = r"models\a2_t1_checkpoints\snapshot_1000000.zip"
A2_BASE_MODEL_PATH = "models/a2_base.zip" # Fallback if training not complete

OPPONENT = "Trained"                      # Defender | Attacker | Hybrid | Follower | Trained | Random | Human
GOAL_SIZE = 64.0                        # Goal half-height in physics units
DETERMINISTIC = True                    # Use deterministic policy actions

# Display labels (auto-derived from model paths)
_stem = lambda p: os.path.splitext(os.path.basename(p))[0] if p else None
if MODE == "A1":
    AGENT_LABEL = _stem(A1_MODEL_PATH) or "A1"
    OPP_LABEL   = _stem(A1_OPP_MODEL_PATH) or "OPP"
else:
    if PLAY_AS == "T1":
        AGENT_LABEL = "A2_T1"
        OPP_LABEL = "A2_T2"
    else:
        AGENT_LABEL = "A2_T2"
        OPP_LABEL = "A2_T1"
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pygame
except ModuleNotFoundError:
    print("pygame not found. Install with:  pip install pygame")
    sys.exit(1)

from stable_baselines3 import PPO
from training.env import HaxballCurriculumEnv, DIR_MAP, KICK_STR, POLE_R, _dist_to_goal_segment, Disc, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP

class PlayFixEnv(HaxballCurriculumEnv):
    def _reset_positions(self):
        super()._reset_positions()
        
        # In A2v1/A2v2, just use the parent's default positioning (random across full half)
        if self.phase in ('A2.0', 'A2v2'):
            return
            
        # 100% Own goal situation (A1 only)
        own_goal_x = -self.HW * self._attack_sign
        own_goal_y = float(self._rng.uniform(-self.goal_y, self.goal_y))
        
        # Mọi player được spawn ngẫu nhiên trong đúng nửa sân của mình
        if self.team_id == 1:
            # Agent is RED (Left half), Opp is BLUE (Right half)
            ax = float(self._rng.uniform(-self.HW * 0.8 + PLYR_R, -PLYR_R))
            ox = float(self._rng.uniform(PLYR_R, self.HW * 0.8 - PLYR_R))
        else:
            # Agent is BLUE (Right half), Opp is RED (Left half)
            ax = float(self._rng.uniform(PLYR_R, self.HW * 0.8 - PLYR_R))
            ox = float(self._rng.uniform(-self.HW * 0.8 + PLYR_R, -PLYR_R))
            
        ay = float(self._rng.uniform(-self.goal_y, self.goal_y))
        oy = float(self._rng.uniform(-self.goal_y, self.goal_y))
        
        dx = own_goal_x - ax
        dy = own_goal_y - ay
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            ndx, ndy = dx / dist, dy / dist
        else:
            ndx, ndy = 1.0, 0.0
            
        gap = float(self._rng.uniform(0.5, 2.0))
        b_dist = PLYR_R + BALL_R + gap
        bx = ax + ndx * b_dist
        by = ay + ndy * b_dist
        
        self.ball = Disc(bx, by, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
        
        self.agents = [
            Disc(ax, ay, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP),
            Disc(ox, oy, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)
        ]

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if terminated or truncated:
            agent_score = self.scores[self.team_id - 1]
            # Reduce goal reward from 30 to 10 for A1
            if agent_score > 0 and self.phase == 'A1':
                reward -= 20.0
                
        return obs, reward, terminated, truncated, info

# ── Display constants ──────────────────────────────────────────────────────────
WIN_W      = 1000
WIN_H      = 600
PANEL_H    = 90       # top info panel height
FIELD_H    = WIN_H - PANEL_H - 10

# Colours
C_BG       = (15,  20,  35)
C_FIELD    = (20,  90,  40)
C_FIELD2   = (18,  82,  36)   # alternating stripe
C_LINE     = (255, 255, 255)
C_GOAL_L   = (100, 180, 255)  # blue goal
C_GOAL_R   = (255, 100, 100)  # red goal
C_BALL     = (240, 220,  50)
C_AGENT    = (255,  80,  80)  # RED player (our agent)
C_OPP      = (80,  130, 255)  # BLUE player (opponent)
C_PANEL    = (10,  15,  28)
C_TEXT     = (220, 230, 255)
C_DIM      = (120, 130, 150)
C_HIT      = (255, 255, 100)
C_WIN      = (80,  255, 140)
C_LOSE     = (255,  80,  80)
C_PAUSE    = (255, 200,  50)
C_SHADOW   = (0,    0,    0,  120)   # for shadow blits

FONT_BIG   = None
FONT_MED   = None
FONT_SM    = None


def init_fonts():
    global FONT_BIG, FONT_MED, FONT_SM
    pygame.font.init()
    FONT_BIG = pygame.font.SysFont("Segoe UI", 40, bold=True)
    FONT_MED = pygame.font.SysFont("Segoe UI", 22)
    FONT_SM  = pygame.font.SysFont("Segoe UI", 16)


# ── Coordinate mapping ─────────────────────────────────────────────────────────
def field_to_screen(fx, fy, env, surf_rect):
    """Map physics coordinates → screen pixel coordinates."""
    HW, HH = env.HW, env.HH
    fx_norm = (fx + HW) / (2 * HW)
    fy_norm = (fy + HH) / (2 * HH)
    sx = surf_rect.left + fx_norm * surf_rect.width
    sy = surf_rect.top  + fy_norm * surf_rect.height
    return int(sx), int(sy)


def px_scale(r, env, surf_rect):
    """Scale radius from physics units → pixels (x-axis scale)."""
    return max(int(r / (2 * env.HW) * surf_rect.width), 2)


# ── Rendering ──────────────────────────────────────────────────────────────────
def draw_field(screen, env, surf_rect):
    sr = surf_rect
    pygame.draw.rect(screen, C_FIELD, sr)

    # Alternating vertical stripes (subtle)
    stripe_w = sr.width // 10
    for i in range(10):
        if i % 2 == 1:
            pygame.draw.rect(screen, C_FIELD2,
                             (sr.left + i * stripe_w, sr.top, stripe_w, sr.height))

    HW, HH = env.HW, env.HH
    gcy  = env.goal_center_y
    gy   = env.goal_y

    # Center line
    cx, _ = field_to_screen(0, 0, env, sr)
    pygame.draw.line(screen, (*C_LINE, 80), (cx, sr.top), (cx, sr.bottom), 1)

    # Center circle
    circle_r = px_scale(env.HH * 0.22, env, sr)
    cy_mid   = sr.top + sr.height // 2
    pygame.draw.circle(screen, C_LINE, (cx, cy_mid), circle_r, 1)

    # Goals
    for side, c_goal in [(-HW, C_GOAL_L), (HW, C_GOAL_R)]:
        top_s = field_to_screen(side, gcy - gy, env, sr)
        bot_s = field_to_screen(side, gcy + gy, env, sr)
        goal_w = max(14, px_scale(HW * 0.06, env, sr))
        if side < 0:
            rect = pygame.Rect(sr.left, top_s[1], goal_w, bot_s[1] - top_s[1])
        else:
            rect = pygame.Rect(sr.right - goal_w, top_s[1], goal_w, bot_s[1] - top_s[1])
        goal_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        goal_surf.fill((*c_goal, 60))
        screen.blit(goal_surf, rect)
        pygame.draw.line(screen, c_goal, top_s, bot_s, 3)
        
        # Draw poles
        pole_r = px_scale(POLE_R, env, sr)
        pygame.draw.circle(screen, (200, 200, 200), top_s, pole_r)
        pygame.draw.circle(screen, (0, 0, 0), top_s, pole_r, 1)
        pygame.draw.circle(screen, (200, 200, 200), bot_s, pole_r)
        pygame.draw.circle(screen, (0, 0, 0), bot_s, pole_r, 1)

    # Field border
    pygame.draw.rect(screen, C_LINE, sr, 2)


def draw_players(screen, env, surf_rect):
    HW = env.HW
    labels = []

    for i, ag in enumerate(env.agents):
        sx, sy = field_to_screen(ag.x, ag.y, env, surf_rect)
        r = max(px_scale(ag.radius, env, surf_rect), 6)

        if i == 0:
            # Agent: determine colour by team
            color = C_AGENT if env.team_id == 1 else C_OPP
            label = AGENT_LABEL
        elif env.phase in ('A2.0', 'A2v2') and env.n_agents == 2 and i == 1:
            # Teammate
            color = C_AGENT if env.team_id == 1 else C_OPP
            label = "TM"
        else:
            # Opponent
            color = C_OPP if env.team_id == 1 else C_AGENT
            label = OPP_LABEL if env.opponent_type == 'Trained' else (env.opponent_type[:3] if env.opponent_type else "BOT")

        # Shadow
        shadow_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(shadow_surf, (0, 0, 0, 80), (r + 2, r + 4), r)
        screen.blit(shadow_surf, (sx - r - 2, sy - r - 2))

        # Body
        pygame.draw.circle(screen, color, (sx, sy), r)
        pygame.draw.circle(screen, C_LINE, (sx, sy), r, 2)

        # Direction indicator (velocity arrow)
        spd = math.hypot(ag.xs, ag.ys)
        if spd > 0.05:
            nx, ny = ag.xs / spd, ag.ys / spd
            scale = r / (2 * HW) * surf_rect.width * 3
            tx = int(sx + nx * scale)
            ty = int(sy + ny * scale)
            pygame.draw.line(screen, C_LINE, (sx, sy), (tx, ty), 2)

        # Label
        lbl = FONT_SM.render(label, True, (255, 255, 255))
        screen.blit(lbl, (sx - lbl.get_width() // 2, sy - lbl.get_height() // 2))
        labels.append((sx, sy, r))


def draw_ball(screen, env, surf_rect):
    b = env.ball
    sx, sy = field_to_screen(b.x, b.y, env, surf_rect)
    r = max(px_scale(b.radius, env, surf_rect), 4)

    # Shadow
    shad = pygame.Surface((r * 2 + 6, r * 2 + 6), pygame.SRCALPHA)
    pygame.draw.circle(shad, (0, 0, 0, 90), (r + 3, r + 6), r)
    screen.blit(shad, (sx - r - 3, sy - r - 3))

    pygame.draw.circle(screen, C_BALL, (sx, sy), r)
    pygame.draw.circle(screen, (180, 140, 20), (sx, sy), r, 2)

    # Speed trail
    spd = math.hypot(b.xs, b.ys)
    if spd > 0.3:
        HW = env.HW
        scale_factor = (r / (2 * HW)) * surf_rect.width
        tx = int(sx - b.xs / spd * r * 2.5)
        ty = int(sy - b.ys / spd * r * 2.5)
        trail_surf = pygame.Surface((abs(tx - sx) * 2 + 4, abs(ty - sy) * 2 + 4), pygame.SRCALPHA)
        pygame.draw.line(screen, (*C_BALL, 60), (sx, sy), (tx, ty), max(1, r // 2))


def draw_panel(screen, env, step, ep_reward, step_reward, episode, total_eps,
               result_str, paused, opp_type):
    panel_rect = pygame.Rect(0, 0, WIN_W, PANEL_H)
    pygame.draw.rect(screen, C_PANEL, panel_rect)
    pygame.draw.line(screen, (50, 60, 80), (0, PANEL_H - 1), (WIN_W, PANEL_H - 1), 1)

    team_str  = "RED" if env.team_id == 1 else "BLUE"
    atk_str   = ">> RIGHT" if env._attack_sign == 1 else "<< LEFT"
    score_str = f"RED {env.scores[0]} - {env.scores[1]} BLUE"

    # Score (big)
    score_surf = FONT_BIG.render(score_str, True, C_TEXT)
    screen.blit(score_surf, (WIN_W // 2 - score_surf.get_width() // 2, 8))

    # Left info
    left_lines = [
        f"Episode {episode + 1}/{total_eps}",
        f"Step {step}/{env.max_steps}",
        f"Ep reward: {ep_reward:+.2f}",
    ]
    for i, line in enumerate(left_lines):
        surf = FONT_SM.render(line, True, C_DIM)
        screen.blit(surf, (14, 8 + i * 18))

    # Right info
    right_lines = [
        f"Team: {team_str}  {atk_str}",
        f"Opponent: {opp_type}",
        f"goal_y: {env.goal_y:.0f}  ep_type: {env.episode_type}",
    ]
    for i, line in enumerate(right_lines):
        surf = FONT_SM.render(line, True, C_DIM)
        screen.blit(surf, (WIN_W - surf.get_width() - 14, 8 + i * 18))

    # Step reward bar (center-bottom of panel)
    rew_color = C_WIN if step_reward > 0.0001 else (C_LOSE if step_reward < -0.0001 else C_DIM)
    rew_str = f"Step rew: {step_reward:+.5f}"
    rew_surf = FONT_SM.render(rew_str, True, rew_color)
    screen.blit(rew_surf, (WIN_W // 2 - rew_surf.get_width() // 2, PANEL_H - 44))

    # Result flash
    if result_str:
        color = C_WIN if "WIN" in result_str or "GOAL" in result_str else C_LOSE
        if "TIMEOUT" in result_str:
            color = C_PAUSE
        rf = FONT_MED.render(result_str, True, color)
        screen.blit(rf, (WIN_W // 2 - rf.get_width() // 2, PANEL_H - 26))

    # Paused
    if paused:
        ps = FONT_MED.render("PAUSED  [SPACE to resume]", True, C_PAUSE)
        screen.blit(ps, (WIN_W // 2 - ps.get_width() // 2, PANEL_H - 26))

    # Controls hint
    hint = "SPACE=pause  R=restart  Q/ESC=quit"
    if env.opponent_type == "Human":
        hint += "   |   OPP: arrows=move  Enter/RCtrl=kick"
    controls = FONT_SM.render(hint, True, (60, 70, 90))
    screen.blit(controls, (WIN_W - controls.get_width() - 10, WIN_H - 18))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    try:
        import numpy.core.numeric
        sys.modules['numpy._core'] = sys.modules['numpy.core']
        sys.modules['numpy._core.numeric'] = sys.modules['numpy.core.numeric']
        sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray', sys.modules['numpy.core'])
    except Exception:
        pass

    if MODE == "A1":
        print(f"Loading model: {A1_MODEL_PATH}")
        main_model = PPO.load(A1_MODEL_PATH, device="cpu")
        env = PlayFixEnv(phase="A1")
        
        if OPPONENT.lower() == "trained" and A1_OPP_MODEL_PATH:
            print(f"Loading opponent model: {A1_OPP_MODEL_PATH}")
            env.opponent_policy = PPO.load(A1_OPP_MODEL_PATH, device="cpu")
            
    elif MODE == "A2v1":
        n_agents = 2 if PLAY_AS == "T1" else 1
        env = PlayFixEnv(phase="A2.0", n_agents=n_agents)
        
        agent_path = A2_T1_MODEL_PATH if PLAY_AS == "T1" else A2_T2_MODEL_PATH
        opp_path   = A2_T2_MODEL_PATH if PLAY_AS == "T1" else A2_T1_MODEL_PATH
        
        if not os.path.exists(agent_path):
            print(f"WARNING: {agent_path} not found. Falling back to base model {A2_BASE_MODEL_PATH}")
            agent_path = A2_BASE_MODEL_PATH
        if not os.path.exists(opp_path):
            print(f"WARNING: {opp_path} not found. Falling back to base model {A2_BASE_MODEL_PATH}")
            opp_path = A2_BASE_MODEL_PATH
            
        print(f"Loading main agent model: {agent_path}")
        main_model = PPO.load(agent_path, device="cpu")
        
        if n_agents == 2:
            env.teammate_policy = main_model
            
        if OPPONENT.lower() == "trained":
            print(f"Loading opponent model: {opp_path}")
            env.opponent_policy = PPO.load(opp_path, device="cpu")
            
    else: # A2v2
        n_agents = 2
        env = PlayFixEnv(phase="A2v2", n_agents=n_agents)
        
        agent_path = A2_T1_MODEL_PATH if PLAY_AS == "T1" else A2_T2_MODEL_PATH
        opp_path   = A2_T2_MODEL_PATH if PLAY_AS == "T1" else A2_T1_MODEL_PATH
        
        if not os.path.exists(agent_path):
            print(f"WARNING: {agent_path} not found. Falling back to base model {A2_BASE_MODEL_PATH}")
            agent_path = A2_BASE_MODEL_PATH
        if not os.path.exists(opp_path):
            print(f"WARNING: {opp_path} not found. Falling back to base model {A2_BASE_MODEL_PATH}")
            opp_path = A2_BASE_MODEL_PATH
            
        print(f"Loading main agent model: {agent_path}")
        main_model = PPO.load(agent_path, device="cpu")
        
        env.teammate_policy = main_model
            
        if OPPONENT.lower() == "trained":
            print(f"Loading opponent model: {opp_path}")
            env.opponent_policy = PPO.load(opp_path, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"Haxball — {MODE} Play")
    clock = pygame.time.Clock()
    init_fonts()

    # Field surface rect (below panel, with 5px margin)
    field_rect = pygame.Rect(5, PANEL_H + 5, WIN_W - 10, FIELD_H)

    # ── Set forced_opponent_type BEFORE first reset so it sticks every episode ─
    if OPPONENT:
        opp_lower = OPPONENT.lower()
        if opp_lower == "human":
            env.forced_opponent_type = "Human"
        elif opp_lower not in ("random", "none", "solo"):
            env.forced_opponent_type = OPPONENT

    is_human_opp = OPPONENT and OPPONENT.lower() == "human"

    # ── Helper: read human opponent keyboard input ────────────────────────────
    def get_human_action():
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:  dx -= 1
        if keys[pygame.K_RIGHT]: dx += 1
        if keys[pygame.K_UP]:    dy -= 1
        if keys[pygame.K_DOWN]:  dy += 1
        kick = 1 if (keys[pygame.K_RETURN] or keys[pygame.K_RCTRL]
                     or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 0
        return (float(dx), float(dy), kick)

    def do_reset():
        env.reset()
        env.team_id      = 1
        env._flip        = +1.0
        env._attack_sign = +1
        env._reset_positions()
        import math as _math
        a = env.agents[0]
        env._prev_dist_to_ball = _math.hypot(a.x - env.ball.x, a.y - env.ball.y)
        env._min_dist_to_ball = env._prev_dist_to_ball
        goal_x = env.HW * env._attack_sign
        # Use segment formula consistent with the current training phase
        env._prev_ball_dist_to_goal = _dist_to_goal_segment(
            env.ball.x, env.ball.y,
            goal_x, env.goal_y, 0.9 * env.goal_y
        )
        env._prev_ball_speed = _math.hypot(env.ball.xs, env.ball.ys)
        env.last_touch = None
        env.step_count = 0
        env.scores = [0, 0]
        return env._get_obs()

    # ── Physics timing ────────────────────────────────────────────────────────
    # Each env.step() = frame_skip=3 ticks @ 60Hz → 50ms per step = 20 steps/s
    RENDER_FPS    = 60
    STEP_INTERVAL = 1.0 / 20.0   # 20 steps/s  (3 ticks × 60Hz = correct real-time)

    paused          = False
    running         = True
    episode         = 0
    result_flash    = ""
    flash_until     = 0.0        # absolute time when flash expires
    post_done_until = 0.0        # wait this long before resetting after match ends
    waiting_reset   = False

    obs        = do_reset()
    ep_reward  = 0.0
    step_reward = 0.0
    step_cnt   = 0
    last_step_t = time.time()

    while running:
        now = time.time()

        # ── Events ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs           = do_reset()
                    ep_reward     = 0.0
                    step_reward   = 0.0
                    step_cnt      = 0
                    result_flash  = ""
                    flash_until   = 0.0
                    waiting_reset = False
                    last_step_t   = time.time()
                elif event.key == pygame.K_1:
                    print("Switching mode to A1")
                    sys.argv = [sys.argv[0]] # we could implement dynamic mode switch but simpler to just edit config
                
        if not running:
            break

        # ── Auto-reset after match ends ──────────────────────────────────────
        if waiting_reset:
            if now >= post_done_until:
                obs           = do_reset()
                ep_reward     = 0.0
                step_reward   = 0.0
                step_cnt      = 0
                result_flash  = ""
                flash_until   = 0.0
                waiting_reset = False
                last_step_t   = time.time()
                episode      += 1
            # Draw frozen state while waiting
            screen.fill(C_BG)
            draw_field(screen, env, field_rect)
            draw_players(screen, env, field_rect)
            draw_ball(screen, env, field_rect)
            draw_panel(screen, env, step_cnt, ep_reward, step_reward, episode, episode + 1,
                       result_flash, False, env.opponent_type)
            pygame.display.flip()
            clock.tick(RENDER_FPS)
            continue

        if paused:
            screen.fill(C_BG)
            draw_field(screen, env, field_rect)
            draw_players(screen, env, field_rect)
            draw_ball(screen, env, field_rect)
            draw_panel(screen, env, step_cnt, ep_reward, step_reward, episode, episode + 1,
                       result_flash if now < flash_until else "", True, env.opponent_type)
            pygame.display.flip()
            clock.tick(RENDER_FPS)
            continue

        # ── Physics step (time-gated to 10 Hz = real HaxBall speed) ─────────
        if now - last_step_t >= STEP_INTERVAL:
            if is_human_opp:
                env.human_opponent_action = get_human_action()
            action, _ = main_model.predict(obs, deterministic=DETERMINISTIC)
            obs, reward, terminated, truncated, _ = env.step(action)
            step_reward = reward
            ep_reward  += reward
            step_cnt   += 1
            last_step_t = now

            done = terminated or truncated
            if done:
                if terminated:
                    result_flash = f"GOAL!  ep_rew={ep_reward:+.2f}"
                else:
                    result_flash = f"TIMEOUT  ep_rew={ep_reward:+.2f}"
                flash_until     = now + 10.0   # show flash until reset
                post_done_until = now + 2.0    # auto-reset after 2 s
                waiting_reset   = True

        # ── Render (always 60 fps) ───────────────────────────────────────────
        screen.fill(C_BG)
        draw_field(screen, env, field_rect)
        draw_players(screen, env, field_rect)
        draw_ball(screen, env, field_rect)
        draw_panel(screen, env, step_cnt, ep_reward, step_reward, episode, episode + 1,
                   result_flash if now < flash_until else "", paused, env.opponent_type)
        pygame.display.flip()
        clock.tick(RENDER_FPS)

    pygame.quit()
    print("Eval done.")


if __name__ == "__main__":
    main()

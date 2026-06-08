"""
play_a3.2.py — Watch A2.2_2 Dynamic Play

Controls (while window is open):
    SPACE   — pause / resume
    R       — restart episode immediately
    Q / ESC — quit

Dynamic A2.2_2 layout:
    Matches randomly alternate between 1v1, 2v1, and 2v2 
    based on the A2.2_2 curriculum probabilities.
"""

import math
import sys
import time
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Configuration ──────────────────────────────────────────────────────────────
AGENT_MODEL_PATH    = "models/2v2 oriented/a3_2_checkpoints/emergency_nopool_4016700.zip"  # Agent + đồng đội
OPPONENT_MODEL_PATH = "models/2v2 oriented/a3_1_checkpoints/snapshot_2750000.zip"  # Đối thủ
SPEED        = 2.0      # Speed multiplier
DETERMINISTIC = True    # Dùng deterministic policy
# ──────────────────────────────────────────────────────────────────────────────

try:
    import pygame
except ModuleNotFoundError:
    print("pygame not found.  pip install pygame")
    sys.exit(1)

from stable_baselines3 import PPO
from training.env import HaxballCurriculumEnv, POLE_R

# ── Display constants ──────────────────────────────────────────────────────────
WIN_W   = 1100
WIN_H   = 640
PANEL_H = 90
FIELD_H = WIN_H - PANEL_H - 10

C_BG     = (15,  20,  35)
C_FIELD  = (20,  90,  40)
C_FIELD2 = (18,  82,  36)
C_LINE   = (255, 255, 255)
C_GOAL_L = (100, 180, 255)
C_GOAL_R = (255, 100, 100)
C_BALL   = (240, 220,  50)
C_RED    = (255,  80,  80)
C_BLUE   = ( 80, 130, 255)
C_PANEL  = ( 10,  15,  28)
C_TEXT   = (220, 230, 255)
C_DIM    = (120, 130, 150)
C_WIN    = ( 80, 255, 140)
C_LOSE   = (255,  80,  80)
C_PAUSE  = (255, 200,  50)

FONT_BIG = FONT_MED = FONT_SM = None


def init_fonts():
    global FONT_BIG, FONT_MED, FONT_SM
    pygame.font.init()
    FONT_BIG = pygame.font.SysFont("Segoe UI", 40, bold=True)
    FONT_MED = pygame.font.SysFont("Segoe UI", 22)
    FONT_SM  = pygame.font.SysFont("Segoe UI", 16)


# ── Coordinate helpers ─────────────────────────────────────────────────────────
def field_to_screen(fx, fy, env, sr):
    HW, HH = env.HW, env.HH
    sx = sr.left + (fx + HW) / (2 * HW) * sr.width
    sy = sr.top  + (fy + HH) / (2 * HH) * sr.height
    return int(sx), int(sy)


def px_scale(r, env, sr):
    return max(int(r / (2 * env.HW) * sr.width), 2)


# ── Rendering ──────────────────────────────────────────────────────────────────
def draw_field(screen, env, sr):
    pygame.draw.rect(screen, C_FIELD, sr)
    sw = sr.width // 10
    for i in range(10):
        if i % 2 == 1:
            pygame.draw.rect(screen, C_FIELD2, (sr.left + i * sw, sr.top, sw, sr.height))

    HW, HH = env.HW, env.HH
    gcy, gy = env.goal_center_y, env.goal_y

    cx, _ = field_to_screen(0, 0, env, sr)
    pygame.draw.line(screen, (*C_LINE, 80), (cx, sr.top), (cx, sr.bottom), 1)
    cy_mid = sr.top + sr.height // 2
    pygame.draw.circle(screen, C_LINE, (cx, cy_mid), px_scale(HH * 0.22, env, sr), 1)

    for side, c_goal in [(-HW, C_GOAL_L), (HW, C_GOAL_R)]:
        top_s = field_to_screen(side, gcy - gy, env, sr)
        bot_s = field_to_screen(side, gcy + gy, env, sr)
        gw = max(14, px_scale(HW * 0.06, env, sr))
        if side < 0:
            rect = pygame.Rect(sr.left, top_s[1], gw, bot_s[1] - top_s[1])
        else:
            rect = pygame.Rect(sr.right - gw, top_s[1], gw, bot_s[1] - top_s[1])
        gs = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        gs.fill((*c_goal, 60))
        screen.blit(gs, rect)
        pygame.draw.line(screen, c_goal, top_s, bot_s, 3)
        pr = px_scale(POLE_R, env, sr)
        for pt in (top_s, bot_s):
            pygame.draw.circle(screen, (200, 200, 200), pt, pr)
            pygame.draw.circle(screen, (0, 0, 0), pt, pr, 1)

    pygame.draw.rect(screen, C_LINE, sr, 2)


def draw_players(screen, env, sr):
    for i, ag in enumerate(env.agents):
        sx, sy = field_to_screen(ag.x, ag.y, env, sr)
        r = max(px_scale(ag.radius, env, sr), 6)
        
        is_teammate = env.agents_team[i] == env.team_id if hasattr(env, 'agents_team') and len(env.agents_team) > i else (i == 0)
        team_of_agent = env.agents_team[i] if hasattr(env, 'agents_team') and len(env.agents_team) > i else (env.team_id if i == 0 else 3 - env.team_id)
        
        color = C_RED if team_of_agent == 1 else C_BLUE

        if i == 0:
            label = "A"
        elif is_teammate:
            label = "TM"
        else:
            opp_idx = i - env.agents_team.count(env.team_id)
            label = f"O{opp_idx + 1}"

        # Shadow
        sh = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(sh, (0, 0, 0, 80), (r + 2, r + 4), r)
        screen.blit(sh, (sx - r - 2, sy - r - 2))

        pygame.draw.circle(screen, color, (sx, sy), r)
        pygame.draw.circle(screen, C_LINE, (sx, sy), r, 2)

        # Velocity arrow
        spd = math.hypot(ag.xs, ag.ys)
        if spd > 0.05:
            nx, ny = ag.xs / spd, ag.ys / spd
            scale = r / (2 * env.HW) * sr.width * 3
            pygame.draw.line(screen, C_LINE, (sx, sy),
                             (int(sx + nx * scale), int(sy + ny * scale)), 2)

        lbl = FONT_SM.render(label, True, (255, 255, 255))
        screen.blit(lbl, (sx - lbl.get_width() // 2, sy - lbl.get_height() // 2))


def draw_ball(screen, env, sr):
    b = env.ball
    sx, sy = field_to_screen(b.x, b.y, env, sr)
    r = max(px_scale(b.radius, env, sr), 4)

    sh = pygame.Surface((r * 2 + 6, r * 2 + 6), pygame.SRCALPHA)
    pygame.draw.circle(sh, (0, 0, 0, 90), (r + 3, r + 6), r)
    screen.blit(sh, (sx - r - 3, sy - r - 3))

    pygame.draw.circle(screen, C_BALL, (sx, sy), r)
    pygame.draw.circle(screen, (180, 140, 20), (sx, sy), r, 2)

    spd = math.hypot(b.xs, b.ys)
    if spd > 0.3:
        tx = int(sx - b.xs / spd * r * 2.5)
        ty = int(sy - b.ys / spd * r * 2.5)
        pygame.draw.line(screen, (*C_BALL, 60), (sx, sy), (tx, ty), max(1, r // 2))


def draw_panel(screen, env, step_cnt, ep_reward, step_reward, episode, result_str, paused):
    panel_rect = pygame.Rect(0, 0, WIN_W, PANEL_H)
    pygame.draw.rect(screen, C_PANEL, panel_rect)
    pygame.draw.line(screen, (50, 60, 80), (0, PANEL_H - 1), (WIN_W, PANEL_H - 1), 1)

    score_str = f"RED {env.scores[0]}  —  {env.scores[1]} BLUE"
    sc = FONT_BIG.render(score_str, True, C_TEXT)
    screen.blit(sc, (WIN_W // 2 - sc.get_width() // 2, 8))

    left_lines = [
        f"Episode {episode + 1}",
        f"Step {step_cnt}/{env.max_steps}",
        f"Ep reward: {ep_reward:+.3f}",
    ]
    for i, ln in enumerate(left_lines):
        s = FONT_SM.render(ln, True, C_DIM)
        screen.blit(s, (14, 8 + i * 18))

    team_str = "RED" if env.team_id == 1 else "BLUE"
    atk_str  = ">> RIGHT" if env._attack_sign == 1 else "<< LEFT"
    
    # Identify mode dynamically based on agents in env
    num_tm = env.agents_team.count(env.team_id)
    num_opp = len(env.agents) - num_tm
    mode_str = f"{num_tm}v{num_opp}"
    
    right_lines = [
        f"Agent team: {team_str}  {atk_str}",
        f"goal_y: {env.goal_y:.0f}  |  A2.2_2  {mode_str}",
        f"Model: {os.path.basename(AGENT_MODEL_PATH)}",
    ]
    for i, ln in enumerate(right_lines):
        s = FONT_SM.render(ln, True, C_DIM)
        screen.blit(s, (WIN_W - s.get_width() - 14, 8 + i * 18))

    rew_color = C_WIN if step_reward > 0.0001 else (C_LOSE if step_reward < -0.0001 else C_DIM)
    rs = FONT_SM.render(f"Step rew: {step_reward:+.5f}", True, rew_color)
    screen.blit(rs, (WIN_W // 2 - rs.get_width() // 2, PANEL_H - 44))

    if result_str:
        color = C_WIN if ("WIN" in result_str or "GOAL" in result_str) else C_LOSE
        if "TIMEOUT" in result_str:
            color = C_PAUSE
        rf = FONT_MED.render(result_str, True, color)
        screen.blit(rf, (WIN_W // 2 - rf.get_width() // 2, PANEL_H - 26))

    if paused:
        ps = FONT_MED.render("PAUSED  [SPACE to resume]", True, C_PAUSE)
        screen.blit(ps, (WIN_W // 2 - ps.get_width() // 2, PANEL_H - 26))

    hint = FONT_SM.render("SPACE=pause  R=restart  Q/ESC=quit", True, (60, 70, 90))
    screen.blit(hint, (WIN_W - hint.get_width() - 10, WIN_H - 18))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"[A2.2_2] Loading agent model  : {AGENT_MODEL_PATH}")
    agent_model = PPO.load(AGENT_MODEL_PATH, device="cpu")

    print(f"[A2.2_2] Loading opponent model: {OPPONENT_MODEL_PATH}")
    opp_model = PPO.load(OPPONENT_MODEL_PATH, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Haxball — A2.2_2 Play (Dynamic 1v1/2v1/2v2)")
    clock = pygame.time.Clock()
    init_fonts()

    field_rect = pygame.Rect(5, PANEL_H + 5, WIN_W - 10, FIELD_H)

    # A2.2_2: phase='A2.2_2', fixed_mode=None → Tự động chọn 1v1, 2v1, 2v2 theo xác suất
    # Since we are not passing opponent_pool, N_a3_2 will be 0 internally. 
    # This means probability will be 10% 1v1 and 90% 2v1 (early training phase emulation).
    env = HaxballCurriculumEnv(phase="A2.2_2", fixed_mode=None)
    env.forced_opponent_type = "Trained"
    env.opponent_type = "Trained"

    # Đồng đội dùng cùng model với agent
    env.current_model = agent_model
    env.opponent_deterministic = DETERMINISTIC

    # Tối đa 2 đối thủ trong A2.2_2 (2v1 hoặc 2v2)
    env.opponent_policies = [opp_model, opp_model]

    # ── Reset helper ─────────────────────────────────────────────────────────
    def do_reset():
        obs, _ = env.reset()
        env.current_model = agent_model
        env.opponent_policies = [opp_model, opp_model]
        env.opponent_deterministic = DETERMINISTIC
        env.step_count = 0
        env.scores = [0, 0]
        return obs

    # ── Timing ────────────────────────────────────────────────────────────────
    RENDER_FPS    = 60
    STEP_INTERVAL = (1.0 / 20.0) / SPEED   # frame_skip=3 @ 60Hz → 20 steps/s

    paused          = False
    running         = True
    episode         = 0
    result_flash    = ""
    flash_until     = 0.0
    post_done_until = 0.0
    waiting_reset   = False

    obs         = do_reset()
    ep_reward   = 0.0
    step_reward = 0.0
    step_cnt    = 0
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

        if not running:
            break

        # ── Auto-reset ───────────────────────────────────────────────────────
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

            screen.fill(C_BG)
            draw_field(screen, env, field_rect)
            draw_players(screen, env, field_rect)
            draw_ball(screen, env, field_rect)
            draw_panel(screen, env, step_cnt, ep_reward, step_reward,
                       episode, result_flash, False)
            pygame.display.flip()
            clock.tick(RENDER_FPS)
            continue

        if paused:
            screen.fill(C_BG)
            draw_field(screen, env, field_rect)
            draw_players(screen, env, field_rect)
            draw_ball(screen, env, field_rect)
            draw_panel(screen, env, step_cnt, ep_reward, step_reward,
                       episode, result_flash if now < flash_until else "", True)
            pygame.display.flip()
            clock.tick(RENDER_FPS)
            continue

        # ── Physics step ─────────────────────────────────────────────────────
        if now - last_step_t >= STEP_INTERVAL:
            action, _ = agent_model.predict(obs, deterministic=DETERMINISTIC)
            obs, reward, terminated, truncated, _ = env.step(action)
            step_reward  = reward
            ep_reward   += reward
            step_cnt    += 1
            last_step_t  = now

            if terminated or truncated:
                agent_score = env.scores[env.team_id - 1]
                opp_score   = env.scores[2 - env.team_id]
                if terminated:
                    if agent_score > opp_score:
                        result_flash = f"WIN!  RED {env.scores[0]} - {env.scores[1]} BLUE  |  ep_rew={ep_reward:+.2f}"
                    elif opp_score > agent_score:
                        result_flash = f"LOSE  RED {env.scores[0]} - {env.scores[1]} BLUE  |  ep_rew={ep_reward:+.2f}"
                    else:
                        result_flash = f"DRAW  RED {env.scores[0]} - {env.scores[1]} BLUE  |  ep_rew={ep_reward:+.2f}"
                else:
                    result_flash = f"TIMEOUT  RED {env.scores[0]} - {env.scores[1]} BLUE  |  ep_rew={ep_reward:+.2f}"
                flash_until     = now + 10.0
                post_done_until = now + 2.5
                waiting_reset   = True

        # ── Render ───────────────────────────────────────────────────────────
        screen.fill(C_BG)
        draw_field(screen, env, field_rect)
        draw_players(screen, env, field_rect)
        draw_ball(screen, env, field_rect)
        draw_panel(screen, env, step_cnt, ep_reward, step_reward,
                   episode, result_flash if now < flash_until else "", paused)
        pygame.display.flip()
        clock.tick(RENDER_FPS)

    pygame.quit()
    print("[A2.2_2] Done.")


if __name__ == "__main__":
    main()

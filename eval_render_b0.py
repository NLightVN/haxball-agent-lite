"""
eval_render_b0.py — Visual evaluation: watch a trained B0 defend-bot defend against A0.

Usage:
    python eval_render_b0.py --b0-model models/b0_best            # B0 vs heuristic A0
    python eval_render_b0.py --b0-model models/b0_best --a0-model models/a0_best
    python eval_render_b0.py --b0-model models/b0_best --speed 3  # 3× faster
    python eval_render_b0.py --b0-model models/b0_best --episodes 20

Controls:
    SPACE   — pause / resume
    +/-     — increase / decrease speed on-the-fly
    R       — restart episode
    Q / ESC — quit
"""

import argparse
import math
import sys
import time

import numpy as np

try:
    import pygame
except ModuleNotFoundError:
    print("pygame not found.  pip install pygame")
    sys.exit(1)

from stable_baselines3 import PPO

from training.env_b0 import (
    HaxballB0Env, predict_ball_trajectory, steps_to_intercept,
    INTERCEPT_RADIUS, PLYR_R, BALL_R,
)

# ── Display constants ───────────────────────────────────────────────────────────
WIN_W   = 1100
WIN_H   = 640
PANEL_H = 100
FIELD_H = WIN_H - PANEL_H - 10

# Colours
C_BG      = (12,  17,  30)
C_FIELD   = (18,  85,  38)
C_FIELD2  = (16,  76,  34)
C_LINE    = (255, 255, 255)
C_GOAL_B0 = (255,  80,  80)   # goal B0 defends — red highlight
C_GOAL_A0 = (80,  140, 255)   # goal A0 attacks toward
C_BALL    = (240, 220,  50)
C_B0      = (80,  220, 120)   # B0 defender — green
C_A0      = (255,  80,  80)   # A0 attacker — red
C_PANEL   = (8,   13,  24)
C_TEXT    = (220, 230, 255)
C_DIM     = (110, 125, 148)
C_WIN     = (80,  255, 140)
C_LOSE    = (255,  80,  80)
C_PAUSE   = (255, 200,  50)

# Ball-line colours
C_TRAJECTORY   = (255, 200,  50, 120)   # ball path (yellow, semi-transparent)
C_TRAJ_DANGER  = (255,  60,  60, 160)   # going into goal_B0 (red)
C_INTERCEPT_PT = (80,  255, 255, 200)   # intercept point (cyan)
C_INTERCEPT_RING = (80, 255, 255, 80)   # intercept radius ring

FONT_BIG = FONT_MED = FONT_SM = None


def init_fonts():
    global FONT_BIG, FONT_MED, FONT_SM
    pygame.font.init()
    FONT_BIG = pygame.font.SysFont("Segoe UI", 36, bold=True)
    FONT_MED = pygame.font.SysFont("Segoe UI", 20)
    FONT_SM  = pygame.font.SysFont("Segoe UI", 15)


# ── Coordinate helpers ──────────────────────────────────────────────────────────
def fts(fx, fy, env, sr):
    """Field physics coords → screen pixel."""
    HW, HH = env.HW, env.HH
    sx = sr.left + (fx + HW) / (2 * HW) * sr.width
    sy = sr.top  + (fy + HH) / (2 * HH) * sr.height
    return int(sx), int(sy)

def px(r, env, sr):
    return max(int(r / (2 * env.HW) * sr.width), 2)


# ── Draw functions ──────────────────────────────────────────────────────────────
def draw_field(screen, env: HaxballB0Env, sr):
    pygame.draw.rect(screen, C_FIELD, sr)

    # Subtle vertical stripes
    sw = sr.width // 10
    for i in range(10):
        if i % 2 == 1:
            pygame.draw.rect(screen, C_FIELD2, (sr.left + i * sw, sr.top, sw, sr.height))

    HW, HH = env.HW, env.HH
    gcy, gy = env.goal_center_y, env.goal_y

    # Center line
    cx, _ = fts(0, 0, env, sr)
    pygame.draw.line(screen, (*C_LINE, 60), (cx, sr.top), (cx, sr.bottom), 1)

    # Center circle
    cr = px(HH * 0.22, env, sr)
    cy_mid = sr.top + sr.height // 2
    pygame.draw.circle(screen, C_LINE, (cx, cy_mid), cr, 1)

    # Goals — highlight goal_B0 in red, goal_A0 in blue
    goal_B0_x = env._goal_sign * HW
    for side in [-HW, HW]:
        c_goal = C_GOAL_B0 if side == goal_B0_x else C_GOAL_A0
        top_s = fts(side, gcy - gy, env, sr)
        bot_s = fts(side, gcy + gy, env, sr)
        gw = max(14, px(HW * 0.06, env, sr))
        if side < 0:
            rect = pygame.Rect(sr.left, top_s[1], gw, bot_s[1] - top_s[1])
        else:
            rect = pygame.Rect(sr.right - gw, top_s[1], gw, bot_s[1] - top_s[1])
        gs = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        gs.fill((*c_goal, 55))
        screen.blit(gs, rect)
        pygame.draw.line(screen, c_goal, top_s, bot_s, 3)

    pygame.draw.rect(screen, C_LINE, sr, 2)


def draw_ball_line(screen, env: HaxballB0Env, sr):
    """Draw predicted trajectory + intercept point on a single overlay surface."""
    line = env._ball_line
    if not line:
        return

    into_goal = env._ball_line_into_goal
    c_traj = C_TRAJ_DANGER[:3] if into_goal else C_TRAJECTORY[:3]

    # Single shared overlay surface — much faster than one Surface per segment
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

    pts = [fts(x, y, env, sr) for x, y in line]
    n = len(pts)
    for i in range(n - 1):
        alpha = max(25, int(160 * (1 - i / n)))
        pygame.draw.line(overlay, (*c_traj, alpha), pts[i], pts[i + 1], 2)

    screen.blit(overlay, (0, 0))

    # Intercept point
    b0 = env.agents[0]
    sti, best_idx = steps_to_intercept(b0.x, b0.y, line)
    if best_idx >= 0:
        ix, iy = line[best_idx]
        isx, isy = fts(ix, iy, env, sr)
        ring_r = px(INTERCEPT_RADIUS, env, sr)
        ring_surf = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(ring_surf, C_INTERCEPT_RING, (ring_r + 2, ring_r + 2), ring_r)
        screen.blit(ring_surf, (isx - ring_r - 2, isy - ring_r - 2))
        pygame.draw.circle(screen, C_INTERCEPT_PT[:3], (isx, isy), max(5, px(BALL_R, env, sr)), 2)


def draw_players(screen, env: HaxballB0Env, sr):
    for i, ag in enumerate(env.agents):
        sx, sy = fts(ag.x, ag.y, env, sr)
        r = max(px(ag.radius, env, sr), 6)
        color = C_B0 if i == 0 else C_A0
        label = "B0" if i == 0 else "A0"

        # Shadow
        shad = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(shad, (0, 0, 0, 80), (r + 2, r + 5), r)
        screen.blit(shad, (sx - r - 2, sy - r - 2))

        # Body
        pygame.draw.circle(screen, color, (sx, sy), r)
        pygame.draw.circle(screen, C_LINE, (sx, sy), r, 2)

        # Velocity arrow
        spd = math.hypot(ag.xs, ag.ys)
        if spd > 0.05:
            nx, ny = ag.xs / spd, ag.ys / spd
            scale = r * 3.5
            tx = int(sx + nx * scale)
            ty = int(sy + ny * scale)
            pygame.draw.line(screen, C_LINE, (sx, sy), (tx, ty), 2)

        # Label
        lbl = FONT_SM.render(label, True, (255, 255, 255))
        screen.blit(lbl, (sx - lbl.get_width() // 2, sy - lbl.get_height() // 2))


def draw_ball_sprite(screen, env: HaxballB0Env, sr):
    b = env.ball
    sx, sy = fts(b.x, b.y, env, sr)
    r = max(px(b.radius, env, sr), 4)

    shad = pygame.Surface((r * 2 + 6, r * 2 + 6), pygame.SRCALPHA)
    pygame.draw.circle(shad, (0, 0, 0, 90), (r + 3, r + 6), r)
    screen.blit(shad, (sx - r - 3, sy - r - 3))

    pygame.draw.circle(screen, C_BALL, (sx, sy), r)
    pygame.draw.circle(screen, (180, 140, 20), (sx, sy), r, 2)

    spd = math.hypot(b.xs, b.ys)
    if spd > 0.3:
        tx = int(sx - b.xs / spd * r * 2.5)
        ty = int(sy - b.ys / spd * r * 2.5)
        pygame.draw.line(screen, (*C_BALL, 60), (sx, sy), (tx, ty), max(1, r // 2))


def draw_panel(screen, env: HaxballB0Env, step, ep_reward, episode,
               goals_saved, goals_conceded, result_str, paused):
    panel_rect = pygame.Rect(0, 0, WIN_W, PANEL_H)
    pygame.draw.rect(screen, C_PANEL, panel_rect)
    pygame.draw.line(screen, (40, 50, 70), (0, PANEL_H - 1), (WIN_W, PANEL_H - 1), 1)

    # Score (big centre)
    score_str = f"Saved: {goals_saved}   Conceded: {goals_conceded}"
    ssf = FONT_BIG.render(score_str, True, C_TEXT)
    screen.blit(ssf, (WIN_W // 2 - ssf.get_width() // 2, 8))

    # Left info
    b0 = env.agents[0]
    sti, _ = steps_to_intercept(b0.x, b0.y, env._ball_line)
    sti_str = f"{sti:.0f}" if sti != float("inf") else "∞"
    team_str = "RED" if env.team_id == 1 else "BLUE"
    goal_side = "LEFT (-HW)" if env._goal_sign < 0 else "RIGHT (+HW)"

    left_lines = [
        f"Episode {episode + 1}  |  Step {step}/{env.max_steps}",
        f"Ep reward: {ep_reward:+.3f}",
        f"Team: {team_str}  |  goal_B0: {goal_side}",
    ]
    for i, line in enumerate(left_lines):
        surf = FONT_SM.render(line, True, C_DIM)
        screen.blit(surf, (14, 6 + i * 18))

    # Right info
    into_str = "⚠ INTO GOAL_B0" if env._ball_line_into_goal else "safe"
    touch_str = env.last_touch or "—"
    right_lines = [
        f"Ball-line: {into_str}",
        f"Steps to intercept: {sti_str}",
        f"Last touch: {touch_str}  |  goal_y: {env.goal_y:.0f}",
    ]
    for i, line in enumerate(right_lines):
        c = (255, 100, 100) if i == 0 and env._ball_line_into_goal else C_DIM
        surf = FONT_SM.render(line, True, c)
        screen.blit(surf, (WIN_W - surf.get_width() - 14, 6 + i * 18))

    # Result flash
    if result_str:
        cflash = C_WIN if "SAVED" in result_str or "TIMEOUT" in result_str else C_LOSE
        rf = FONT_MED.render(result_str, True, cflash)
        screen.blit(rf, (WIN_W // 2 - rf.get_width() // 2, PANEL_H - 24))

    if paused:
        ps = FONT_MED.render("PAUSED  [SPACE]", True, C_PAUSE)
        screen.blit(ps, (WIN_W // 2 - ps.get_width() // 2, PANEL_H - 24))

    hint = FONT_SM.render("SPACE=pause  R=restart  Q/ESC=quit", True, (50, 60, 82))
    screen.blit(hint, (WIN_W - hint.get_width() - 10, WIN_H - 17))


# ── Arg parsing ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--b0-model",  required=True, help="Path to B0 model .zip")
    p.add_argument("--a0-model",  default=None,  help="Path to A0 model .zip (optional)")
    p.add_argument("--episodes",  default=999,   type=int)
    p.add_argument("--speed",     default=1.0,   type=float,
                   help="Playback speed multiplier (default 1.0 = real-time, 3.0 = 3× faster)")
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"Loading B0 model: {args.b0_model}")
    b0_model = PPO.load(args.b0_model, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Haxball — B0 Defend Eval")
    clock  = pygame.time.Clock()
    init_fonts()

    field_rect = pygame.Rect(5, PANEL_H + 5, WIN_W - 10, FIELD_H)

    env = HaxballB0Env(a0_model_path=args.a0_model)

    RENDER_FPS    = 60
    speed         = max(0.1, args.speed)
    STEP_INTERVAL = 1.0 / (10.0 * speed)   # e.g. speed=3 → 30 decisions/s

    paused          = False
    running         = True
    episode         = 0
    goals_saved     = 0
    goals_conceded  = 0
    result_flash    = ""
    flash_until     = 0.0
    post_done_until = 0.0
    waiting_reset   = False

    obs, _ = env.reset()
    ep_reward  = 0.0
    step_cnt   = 0
    last_step_t = time.time()

    while running and episode < args.episodes:
        now = time.time()

        # ── Events ──────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = min(speed * 1.5, 20.0)
                    STEP_INTERVAL = 1.0 / (10.0 * speed)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed / 1.5, 0.1)
                    STEP_INTERVAL = 1.0 / (10.0 * speed)
                elif event.key == pygame.K_r:
                    obs, _       = env.reset()
                    ep_reward    = 0.0
                    step_cnt     = 0
                    result_flash = ""
                    flash_until  = 0.0
                    waiting_reset = False
                    last_step_t  = time.time()

        if not running:
            break

        # ── Auto-reset after episode ─────────────────────────────────────────────
        if waiting_reset:
            if now >= post_done_until:
                obs, _       = env.reset()
                ep_reward    = 0.0
                step_cnt     = 0
                result_flash = ""
                flash_until  = 0.0
                waiting_reset = False
                last_step_t  = time.time()
                episode     += 1
            screen.fill(C_BG)
            draw_field(screen, env, field_rect)
            draw_ball_line(screen, env, field_rect)
            draw_players(screen, env, field_rect)
            draw_ball_sprite(screen, env, field_rect)
            draw_panel(screen, env, step_cnt, ep_reward, episode,
                       goals_saved, goals_conceded, result_flash, False)
            pygame.display.flip()
            clock.tick(RENDER_FPS)
            continue

        if paused:
            screen.fill(C_BG)
            draw_field(screen, env, field_rect)
            draw_ball_line(screen, env, field_rect)
            draw_players(screen, env, field_rect)
            draw_ball_sprite(screen, env, field_rect)
            draw_panel(screen, env, step_cnt, ep_reward, episode,
                       goals_saved, goals_conceded,
                       result_flash if now < flash_until else "", True)
            pygame.display.flip()
            clock.tick(RENDER_FPS)
            continue

        # ── Physics step(s) ───────────────────────────────────────────────────
        # Run as many steps as needed to catch up (important at high speed)
        steps_this_frame = 0
        MAX_STEPS_PER_FRAME = 8  # cap to avoid freezing the render
        while (now - last_step_t >= STEP_INTERVAL
               and not waiting_reset
               and steps_this_frame < MAX_STEPS_PER_FRAME):
            action, _ = b0_model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            step_cnt  += 1
            last_step_t += STEP_INTERVAL   # advance by fixed interval, not wall-clock
            steps_this_frame += 1

            done = terminated or truncated
            if done:
                if terminated:
                    goals_conceded += 1
                    result_flash = f"CONCEDED  (ep_rew={ep_reward:+.2f})"
                else:
                    goals_saved += 1
                    result_flash = f"TIMEOUT — SAVED!  (ep_rew={ep_reward:+.2f})"
                flash_until     = now + 10.0
                post_done_until = now + 2.0
                waiting_reset   = True
                break

        # ── Render ───────────────────────────────────────────────────────────────
        screen.fill(C_BG)
        draw_field(screen, env, field_rect)
        draw_ball_line(screen, env, field_rect)
        draw_players(screen, env, field_rect)
        draw_ball_sprite(screen, env, field_rect)
        draw_panel(screen, env, step_cnt, ep_reward, episode,
                   goals_saved, goals_conceded,
                   result_flash if now < flash_until else "", paused)
        pygame.display.flip()
        clock.tick(RENDER_FPS)

    pygame.quit()
    print(f"\nEval done — {episode} episodes  |  Saved: {goals_saved}  Conceded: {goals_conceded}")


if __name__ == "__main__":
    main()

"""
play_a0.py — Interactive Play for A0
Watch the A0 AI in a pure Python environment.

Controls (while window is open):
    SPACE   — pause / resume
    R       — restart episode immediately
    Q / ESC — quit
"""

import math
import sys
import time
import os

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Configuration ─────────────────────────────────────────────────────────────
A3_MODEL_PATH = r"models\A3_checkpoints\a3_best.zip"  # "random" de xem agent chua train
DETERMINISTIC = True               # Use deterministic policy actions

# Display labels
_stem = lambda p: os.path.splitext(os.path.basename(p))[0] if p else None
AGENT_LABEL = _stem(A3_MODEL_PATH) or "A3"
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pygame
except ModuleNotFoundError:
    print("pygame not found. Install with:  pip install pygame")
    sys.exit(1)

from sb3_contrib import MaskablePPO
from training.envs.env_a3 import A3Env
from training.env import DIR_MAP, KICK_STR, POLE_R

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
C_PANEL    = (10,  15,  28)
C_TEXT     = (220, 230, 255)
C_DIM      = (120, 130, 150)
C_HIT      = (255, 255, 100)
C_WIN      = (80,  255, 140)
C_LOSE     = (255,  80,  80)
C_PAUSE    = (255, 200,  50)

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
    HW, HH = env.HW, env.HH
    fx_norm = (fx + HW) / (2 * HW)
    fy_norm = (fy + HH) / (2 * HH)
    sx = surf_rect.left + fx_norm * surf_rect.width
    sy = surf_rect.top  + fy_norm * surf_rect.height
    return int(sx), int(sy)

def px_scale(r, env, surf_rect):
    return max(int(r / (2 * env.HW) * surf_rect.width), 2)

# ── Rendering ──────────────────────────────────────────────────────────────────
def draw_field(screen, env, surf_rect):
    sr = surf_rect
    pygame.draw.rect(screen, C_FIELD, sr)

    stripe_w = sr.width // 10
    for i in range(10):
        if i % 2 == 1:
            pygame.draw.rect(screen, C_FIELD2,
                             (sr.left + i * stripe_w, sr.top, stripe_w, sr.height))

    HW, HH = env.HW, env.HH
    gcy  = env.goal_center_y
    gy   = env.goal_y

    cx, _ = field_to_screen(0, 0, env, sr)
    pygame.draw.line(screen, (*C_LINE, 80), (cx, sr.top), (cx, sr.bottom), 1)

    circle_r = px_scale(env.HH * 0.22, env, sr)
    cy_mid   = sr.top + sr.height // 2
    pygame.draw.circle(screen, C_LINE, (cx, cy_mid), circle_r, 1)

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
        
        pole_r = px_scale(POLE_R, env, sr)
        pygame.draw.circle(screen, (200, 200, 200), top_s, pole_r)
        pygame.draw.circle(screen, (0, 0, 0), top_s, pole_r, 1)
        pygame.draw.circle(screen, (200, 200, 200), bot_s, pole_r)
        pygame.draw.circle(screen, (0, 0, 0), bot_s, pole_r, 1)

    pygame.draw.rect(screen, C_LINE, sr, 2)

def draw_players(screen, env, surf_rect):
    HW = env.HW
    for i, ag in enumerate(env.agents):
        sx, sy = field_to_screen(ag.x, ag.y, env, surf_rect)
        r = max(px_scale(ag.radius, env, surf_rect), 6)

        if i < 3:
            color = C_AGENT
            label = AGENT_LABEL if i == 0 else f"A{i}"
        else:
            color = (100, 150, 255)  # Blue for opponent
            opp_i = i - 3
            opp_types = getattr(env, '_opp_types', None)
            if opp_types and opp_i < len(opp_types):
                label = str(opp_types[opp_i])[:3].upper()   # e.g. "WAN", "FOL", "RAN"
            else:
                label = "BOT"

        shadow_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(shadow_surf, (0, 0, 0, 80), (r + 2, r + 4), r)
        screen.blit(shadow_surf, (sx - r - 2, sy - r - 2))

        pygame.draw.circle(screen, color, (sx, sy), r)
        pygame.draw.circle(screen, C_LINE, (sx, sy), r, 2)

        spd = math.hypot(ag.xs, ag.ys)
        if spd > 0.05:
            nx, ny = ag.xs / spd, ag.ys / spd
            scale = r / (2 * HW) * surf_rect.width * 3
            tx = int(sx + nx * scale)
            ty = int(sy + ny * scale)
            pygame.draw.line(screen, C_LINE, (sx, sy), (tx, ty), 2)

        lbl = FONT_SM.render(label, True, (255, 255, 255))
        screen.blit(lbl, (sx - lbl.get_width() // 2, sy - lbl.get_height() // 2))

def draw_ball(screen, env, surf_rect):
    b = env.ball
    sx, sy = field_to_screen(b.x, b.y, env, surf_rect)
    r = max(px_scale(b.radius, env, surf_rect), 4)

    shad = pygame.Surface((r * 2 + 6, r * 2 + 6), pygame.SRCALPHA)
    pygame.draw.circle(shad, (0, 0, 0, 90), (r + 3, r + 6), r)
    screen.blit(shad, (sx - r - 3, sy - r - 3))

    pygame.draw.circle(screen, C_BALL, (sx, sy), r)
    pygame.draw.circle(screen, (180, 140, 20), (sx, sy), r, 2)

    spd = math.hypot(b.xs, b.ys)
    if spd > 0.3:
        HW = env.HW
        tx = int(sx - b.xs / spd * r * 2.5)
        ty = int(sy - b.ys / spd * r * 2.5)
        pygame.draw.line(screen, (*C_BALL, 60), (sx, sy), (tx, ty), max(1, r // 2))

def draw_panel(screen, env, step, ep_reward, step_reward, episode, total_eps,
               result_str, paused, infos=None):
    panel_rect = pygame.Rect(0, 0, WIN_W, PANEL_H)
    pygame.draw.rect(screen, C_PANEL, panel_rect)
    pygame.draw.line(screen, (50, 60, 80), (0, PANEL_H - 1), (WIN_W, PANEL_H - 1), 1)

    team_str  = "RED" if env.team_id == 1 else "BLUE"
    atk_str   = ">> RIGHT" if env._attack_sign == 1 else "<< LEFT"
    score_str = f"RED {env.scores[0]} - {env.scores[1]} BLUE"

    score_surf = FONT_BIG.render(score_str, True, C_TEXT)
    screen.blit(score_surf, (WIN_W // 2 - score_surf.get_width() // 2, 8))

    left_lines = [
        f"Episode {episode + 1}/{total_eps}",
        f"Step {step}/{env.max_steps}",
        f"Ep reward: {ep_reward:+.2f}",
    ]
    for i, line in enumerate(left_lines):
        surf = FONT_SM.render(line, True, C_DIM)
        screen.blit(surf, (14, 8 + i * 18))

    right_lines = [
        f"Team: {team_str}  {atk_str}",
        f"Phase: {env.phase}",
        f"goal_y: {env.goal_y:.0f} | No Opponent",
    ]
    if infos and len(infos) > 0:
        sp = infos[0].get("marl/self_pass", [False, False, False])
        if isinstance(sp, list):
            right_lines.append(f"Self-Pass: {sp[0]} | {sp[1]} | {sp[2]}")
        else:
            right_lines.append(f"Self-Pass: {bool(sp)}")
        mul = infos[0].get("marl/mult", [1.0, 1.0, 1.0])
        if isinstance(mul, list):
            right_lines.append(f"Mult: {mul[0]:.2f} | {mul[1]:.2f} | {mul[2]:.2f}")
        else:
            right_lines.append(f"Mult: {mul:.2f}")
            
        adv = infos[0].get("marl/adv_rew", [0.0, 0.0, 0.0])
        bck = infos[0].get("marl/back_pen", [0.0, 0.0, 0.0])
        trn = infos[0].get("marl/turn_pen", [0.0, 0.0, 0.0])
        if isinstance(adv, list):
            right_lines.append(f"Adv: {adv[0]:.4f} | {adv[1]:.4f} | {adv[2]:.4f}")
            right_lines.append(f"Bck: {bck[0]:.4f} | {bck[1]:.4f} | {bck[2]:.4f}")
            right_lines.append(f"Trn: {trn[0]:.4f} | {trn[1]:.4f} | {trn[2]:.4f}")
    for i, line in enumerate(right_lines):
        surf = FONT_SM.render(line, True, C_DIM)
        screen.blit(surf, (WIN_W - surf.get_width() - 14, 8 + i * 18))

    if isinstance(step_reward, list):
        rew_str = f"Step rew: {step_reward[0]:+.5f} | {step_reward[1]:+.5f} | {step_reward[2]:+.5f}"
        rew_color = C_WIN if step_reward[0] > 0.0001 else (C_LOSE if step_reward[0] < -0.0001 else C_DIM)
    else:
        rew_color = C_WIN if step_reward > 0.0001 else (C_LOSE if step_reward < -0.0001 else C_DIM)
        rew_str = f"Step rew: {step_reward:+.5f}"
    
    rew_surf = FONT_SM.render(rew_str, True, rew_color)
    screen.blit(rew_surf, (WIN_W // 2 - rew_surf.get_width() // 2, PANEL_H - 44))

    if result_str:
        color = C_WIN if "WIN" in result_str or "GOAL" in result_str else C_LOSE
        if "TIMEOUT" in result_str:
            color = C_PAUSE
        rf = FONT_MED.render(result_str, True, color)
        screen.blit(rf, (WIN_W // 2 - rf.get_width() // 2, PANEL_H - 26))

    if paused:
        ps = FONT_MED.render("PAUSED  [SPACE to resume]", True, C_PAUSE)
        screen.blit(ps, (WIN_W // 2 - ps.get_width() // 2, PANEL_H - 26))

    hint = "SPACE=pause  R=restart  Q/ESC=quit"
    controls = FONT_SM.render(hint, True, (60, 70, 90))
    screen.blit(controls, (WIN_W - controls.get_width() - 10, WIN_H - 18))

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    env = A3Env()
    env.total_timesteps_elapsed = 5_000_000

    print(f"Loading A3 model: {A3_MODEL_PATH}")
    a3_model = None
    if A3_MODEL_PATH == "random":
        from sb3_contrib import MaskablePPO
        import torch
        print("Khoi tao agent ngau nhien chua train.")
        a3_model = MaskablePPO("MlpPolicy", env, device="cpu")
        with torch.no_grad():
            a3_model.policy.action_net.bias.data[9] += 2.1972
    else:
        try:
            from sb3_contrib import MaskablePPO
            a3_model = MaskablePPO.load(A3_MODEL_PATH, device="cpu")
            print("Da tai model thanh cong. AI se tu dieu khien.")
        except Exception as e:
            print(f"Khong tim thay model {A3_MODEL_PATH}. Chuyen sang che do NGUOI CHOI.")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Haxball — A3 Play (Human Mode)" if a3_model is None else "Haxball — A3 Play")
    clock = pygame.time.Clock()
    init_fonts()

    field_rect = pygame.Rect(5, PANEL_H + 5, WIN_W - 10, FIELD_H)

    def do_reset():
        obs, _ = env.reset()
        return obs

    def get_human_action_idx():
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:  dx -= 1
        if keys[pygame.K_RIGHT]: dx += 1
        if keys[pygame.K_UP]:    dy -= 1  # game Y: up = negative
        if keys[pygame.K_DOWN]:  dy += 1

        # Flip dx so arrow keys match screen direction regardless of which side agent is on
        dx = dx * env._attack_sign
        
        kick = 1 if (keys[pygame.K_RETURN] or keys[pygame.K_LSHIFT]) else 0
        
        if dx == 0 and dy == 0: dir_idx = 0
        elif dx == 1 and dy == 0: dir_idx = 1
        elif dx == -1 and dy == 0: dir_idx = 2
        elif dx == 0 and dy == -1: dir_idx = 3
        elif dx == 0 and dy == 1: dir_idx = 4
        elif dx == 1 and dy == -1: dir_idx = 5
        elif dx == -1 and dy == -1: dir_idx = 6
        elif dx == 1 and dy == 1: dir_idx = 7
        elif dx == -1 and dy == 1: dir_idx = 8
        else: dir_idx = 0
        
        return [dir_idx, kick]

    RENDER_FPS    = 60
    STEP_INTERVAL = 1.0 / 20.0

    paused          = False
    running         = True
    episode         = 0
    result_flash    = ""
    flash_until     = 0.0
    post_done_until = 0.0
    waiting_reset   = False
    infos           = None

    obs        = do_reset()
    ep_reward  = 0.0
    step_reward = 0.0
    step_cnt   = 0
    last_step_t = time.time()

    while running:
        now = time.time()

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
                    infos         = None
                    last_step_t   = time.time()

        if not running:
            break

        if waiting_reset:
            if now >= post_done_until:
                obs           = do_reset()
                ep_reward     = 0.0
                step_reward   = 0.0
                step_cnt      = 0
                result_flash  = ""
                flash_until   = 0.0
                waiting_reset = False
                infos         = None
                last_step_t   = time.time()
                episode      += 1
            screen.fill(C_BG)
            draw_field(screen, env, field_rect)
            draw_players(screen, env, field_rect)
            draw_ball(screen, env, field_rect)
            draw_panel(screen, env, step_cnt, ep_reward, step_reward, episode, episode + 1,
                       result_flash, False, infos)
            pygame.display.flip()
            clock.tick(RENDER_FPS)
            continue

        if paused:
            screen.fill(C_BG)
            draw_field(screen, env, field_rect)
            draw_players(screen, env, field_rect)
            draw_ball(screen, env, field_rect)
            draw_panel(screen, env, step_cnt, ep_reward, step_reward, episode, episode + 1,
                       result_flash if now < flash_until else "", True, infos)
            pygame.display.flip()
            clock.tick(RENDER_FPS)
            continue

        if now - last_step_t >= STEP_INTERVAL:
            if a3_model is None:
                action_0 = get_human_action_idx()
                action = [action_0, [0, 0], [0, 0]]
            else:
                action = []
                for i in range(3):
                    act, _ = a3_model.predict(obs[i], action_masks=env.action_masks(), deterministic=DETERMINISTIC)
                    action.append(act)
                
            obs, reward, terminated, truncated, infos = env.step(action)
            step_reward = reward if isinstance(reward, list) else [reward]*3
            ep_reward  += reward[0] if isinstance(reward, list) else reward
            step_cnt   += 1
            last_step_t = now

            done = terminated or truncated
            if done:
                if terminated:
                    result_flash = f"GOAL!  ep_rew={ep_reward:+.2f}"
                else:
                    result_flash = f"TIMEOUT  ep_rew={ep_reward:+.2f}"
                flash_until     = now + 10.0
                post_done_until = now + 2.0
                waiting_reset   = True

        screen.fill(C_BG)
        draw_field(screen, env, field_rect)
        draw_players(screen, env, field_rect)
        draw_ball(screen, env, field_rect)
        draw_panel(screen, env, step_cnt, ep_reward, step_reward, episode, episode + 1,
                   result_flash if now < flash_until else "", paused, infos)
        pygame.display.flip()
        clock.tick(RENDER_FPS)

    pygame.quit()
    print("Eval done.")

if __name__ == "__main__":
    main()

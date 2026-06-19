"""
HaxballA0Env — Stage A0 Training Environment
============================================
Task : 1 RED agent, 0 opponents.
Goal : Score into the RIGHT goal (+HW) within MAX_STEPS steps.
       Each step = FRAME_SKIP physics ticks @ 60 Hz → ≈ 100 ms / decision.

Field : Randomly sized each episode — sampled from real futsal map presets:
        • 1v1 maps (n_agents=1): winkys-futsal  (368×171, goal_y=64)
        • 2v2 maps (n_agents=2): felon-network-v2 (520×242, goal_y=76)
                                  galaxy-futsal-v1v2 (401×200, goal_y=70)
        goal_y  : curriculum — Phase 0 enlarges goal toward [40%, 60%] × HH,
                  blending back to preset goal_y by GOAL_CURRICULUM_STEPS = 1,000,000 timesteps.

Action space : MultiDiscrete([9, 2])
    dim-0: movement direction (0=stay, 1=R, 2=L, 3=U, 4=D, 5=UR, 6=UL, 7=DR, 8=DL)
    dim-1: kick (0=no, 1=yes)

Observation  : 100-dim float32 (matches AgentAPI.getObs spec in context.md)

Reward shaping (A0.1):
    Dense:
        +Δ_approach_ball_record × 0.003   (new all-time closest dist to ball; skip if already touching)
        ±Δ_ball_to_goal          × 0.002   (ball closer → +, ball farther → −, every step)
        +0.5  (first touch of the ball, one-time)
    Terminal:
        +30.0 (goal scored)
        (timeout: no penalty)
"""

import math
from typing import Optional

import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces

# ─────────────────────────────────────────────────────────────────────────────
# Physics constants (must match test_index.html exactly)
# ─────────────────────────────────────────────────────────────────────────────
BALL_R        = 6.25
BALL_DAMP     = 0.99
BALL_BCOEF    = 0.4
BALL_IMASS    = 1.5

PLYR_R        = 15.0
PLYR_DAMP     = 0.96
PLYR_IMASS    = 0.5
PLYR_BCOEF    = 0.0
PLYR_ACC      = 0.11
PLYR_KICK_ACC = 0.083
KICK_STR      = 5.0
KICK_RANGE    = 4.0   # surface-gap threshold: dist - r_player - r_ball < 4

POLE_R        = 4.0   # Matches standard futsal map pole physics
POLE_BCOEF    = 0.1
POLE_IMASS    = 0.0   # Immovable

OUTER_PAD     = 35.0  # players can go this far outside field lines (y only)
GOAL_DEPTH    = 75.0  # depth of goal channel behind goal line

# ─────────────────────────────────────────────────────────────────────────────
# Observation normalisation constants (matches agent-api.js)
# ─────────────────────────────────────────────────────────────────────────────
NORM      = 800.0
MAX_SPEED = 10.0
DIAG      = math.sqrt(NORM ** 2 + NORM ** 2)   # ≈ 1131.4
N_TM      = 4   # max teammate slots
N_OPP     = 5   # max opponent slots
OBS_DIM = 4 + 7 + 8 + 11 + 2 + 12 + N_TM * 14 + N_OPP * 13  # = 165

# ─────────────────────────────────────────────────────────────────────────────
# Training meta-constants
# ─────────────────────────────────────────────────────────────────────────────
FRAME_SKIP      = 6            # physics ticks per agent decision
PHYSICS_HZ     = 60

TIME_MAX_SECS  = 10 * 60       # 10 minutes — fixed normalisation ceiling
TIME_MAX_STEPS = TIME_MAX_SECS * PHYSICS_HZ // FRAME_SKIP  # = 6000 steps

# Normalisation ceiling for episode scale obs (matches TIME_MAX_STEPS = 10 min)
MAX_STEPS_ALL_MODES = 6000

# Real futsal map presets: (HW, HH, goal_y, size_class)
MAP_PRESETS = [
    (368.0, 171.0, 64.0, '1v1'),   
    (520.0, 242.0, 76.0, '2v2'),   
    (401.0, 200.0, 70.0, '2v2'),
    (620.0, 270.0, 80.0, '3v3'),
]

# Field & Time Curriculum Thresholds
CURRICULUM_PHASE2 =   300_000
CURRICULUM_PHASE3 = 1_000_000
CURRICULUM_TIME_1 = 2_000_000
CURRICULUM_TIME_2 = 3_000_000
CURRICULUM_A0_1_STEPS = 3_000_000

# A1 Dual-Episode Curriculum
# - Precision episode:  small goal (0.3–0.6×), no opponent
# - Opponent episode:   goal shrinks 1.4× → 0.6× over A1_OPP_CURRICULUM_STEPS
A1_PRECISION_RATIO      = 0.00           # 100% opponent episodes
A1_OPP_CURRICULUM_STEPS = 2_000_000      # steps before goal fully narrows
A1_OPP_GOAL_START       = 1.0           # goal scale at A1 t=0
A1_OPP_GOAL_END         = 1.0           # goal scale at t >= A1_OPP_CURRICULUM_STEPS


# ─────────────────────────────────────────────────────────────────────────────
# Direction map: action-dim-0 → (dx, dy)
# ─────────────────────────────────────────────────────────────────────────────
DIR_MAP = np.array([
    [ 0,  0],  # 0 stay
    [ 1,  0],  # 1 right
    [-1,  0],  # 2 left
    [ 0, -1],  # 3 up
    [ 0,  1],  # 4 down
    [ 1, -1],  # 5 up-right
    [-1, -1],  # 6 up-left
    [ 1,  1],  # 7 down-right
    [-1,  1],  # 8 down-left
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
def _dist_to_goal_segment(bx: float, by: float, gx: float, gy_top: float, gy_bot: float) -> float:
    # gy_top is goal_y (positive), gy_bot is -goal_y (negative)
    # The segment is from (gx, gy_bot) to (gx, gy_top)
    if by > gy_top:
        return math.hypot(bx - gx, by - gy_top)
    elif by < gy_bot:
        return math.hypot(bx - gx, by - gy_bot)
    else:
        return abs(bx - gx)

# ─────────────────────────────────────────────────────────────────────────────
class Disc:
    """Mutable physics disc (ball or player)."""
    __slots__ = ['x', 'y', 'xs', 'ys', 'radius', 'imass', 'bcoef', 'damp']

    def __init__(self, x, y, xs, ys, radius, imass, bcoef, damp):
        self.x = float(x); self.y = float(y)
        self.xs = float(xs); self.ys = float(ys)
        self.radius = float(radius)
        self.imass  = float(imass)
        self.bcoef  = float(bcoef)
        self.damp   = float(damp)


# ─────────────────────────────────────────────────────────────────────────────
class HaxballCurriculumEnv(gym.Env):
    """
    Stage A0 & A1 Training Environment.
    A0: 1 RED agent, score into RIGHT goal to reset. 1m time limit. Penalty -0.0002/step.
    A1: 1v1 up to 3 goals. Standard map preset. Touch reward. 3m time limit.
    """

    metadata = {'render_modes': []}

    def __init__(self, phase: str = 'A0', n_agents: int = 1, seed: Optional[int] = None):
        super().__init__()

        self.phase = phase
        self.n_agents = n_agents
        self._rng = np.random.default_rng(seed)
        
        # MARL physics ticks per step
        self.frame_skip = 3

        # Gymnasium spaces
        self.obs_dim = OBS_DIM
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # 1v1 action space. A1 needs opponent action as well if opponent is trained outside.
        # But we'll run the opponent policy inside step() to keep it simple. So action is just agent's action.
        self.action_space = spaces.MultiDiscrete([9, 2])

        # Internal state (populated in reset)
        self.ball: Optional[Disc]    = None
        self.agents: list[Disc]      = [] # Agent 0 is agent, Agent 1 is opp.
        self.HW    = 0.0
        self.HH    = 0.0
        self.goal_y = 0.0
        self.goal_center_y = 0.0
        self.step_count = 0
        self.max_steps = 150 # dynamic based on curriculum
        self.team_id     = 1    # 1=RED (+HW), 2=BLUE (-HW)  — randomized each episode
        self._flip       = 1.0  # +1 RED, -1 BLUE  (flips x in obs)
        self._attack_sign = 1   # +1 RED, -1 BLUE  (for reward direction)
        
        # Track farmed rewards for penalizing at timeout
        self._farmed_aim_reward = 0.0

        self.total_timesteps_elapsed: int = 0
        
        # A1 state
        self.opponent_type = 'None'   # 'Defender', 'Attacker', 'Trained', 'Hybrid', 'None'
        self.opponent_policy = None   # Only set if 'Trained'
        self.episode_type = 'precision'  # 'precision' | 'opponent'
        self.a0_model_path: Optional[str] = None  # set externally before training
        self.scores = [0, 0]          # RED, BLUE
        self.last_touch = None
        self.last_touch_team = None
        
        # MARL Reward Logic
        self.investment_sequence = []
        self.opp_possession_time = 0.0
        self.dribble_start_time = 0.0
        self.self_pass_shooter = None
        self.self_pass_active = False
        self.real_pass_active = False        # 'A' for Agent, 'O' for Opponent
        self._hybrid_mode = 'follower'  # 'follower' | 'defender' (Hybrid bot state)
        self.human_opponent_action = (0.0, 0.0, 0)  # set externally by eval_render each frame
        self.forced_opponent_type: Optional[str] = None  # if set, bypass random selection every episode
        self._a0_1_best_ball_dist_to_goal = 0.0
        self._a0_1_steps_since_ball_record = 0
        self._a0_1_record_ball_to_goal_reward_total = 0.0

        # Previous-step tracking for dense rewards
        self._prev_dist_to_ball = 0.0
        self._prev_ball_dist_to_goal = 0.0
        self._prev_ball_x       = 0.0
        self._prev_ball_y       = 0.0
        self._prev_ball_speed   = 0.0
        self.last_action_kick   = 0

    # ─── reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ep_secs = 90 # 1.5 minutes
        if getattr(self, 'phase', '') in ['A0', 'A0.1']:
            ep_secs = 15
        self.max_steps = ep_secs * PHYSICS_HZ // self.frame_skip
        
        ts = getattr(self, 'total_timesteps_elapsed', 0)
        if getattr(self, 'phase', '') == 'A3':
            self.max_steps = 1800
        elif getattr(self, 'phase', '') == 'A0.1':
            self.max_steps = 300
        else:
            if 2_000_000 <= ts < 4_000_000:
                self.max_steps = 300
            elif ts >= 4_000_000:
                self.max_steps = 200
        self.step_count = 0
        self.scores = [0, 0]

        self.team_id      = int(self._rng.integers(1, 3))
        self._flip        = 1.0 if self.team_id == 1 else -1.0
        self._attack_sign = 1   if self.team_id == 1 else -1

        self._reset_positions()
        
        a = self.agents[0]
        self._prev_dist_to_ball = math.hypot(a.x - self.ball.x, a.y - self.ball.y)
        goal_x = self.HW * self._attack_sign
        self._prev_ball_dist_to_goal = _dist_to_goal_segment(self.ball.x, self.ball.y, goal_x, self.goal_y, -self.goal_y)
        self._prev_ball_x       = self.ball.x
        self._prev_ball_y       = self.ball.y
        self._prev_ball_speed   = math.hypot(self.ball.xs, self.ball.ys)
        
        self.last_touch = None
        self.last_touch_team = None
        self.last_touch_time = -999.0       # time of last touch event (in seconds)
        self.investment_sequence.clear()
        self.investment_sequence = []
        self.dribble_start_time = 0.0
        self.scores = [0, 0]
        self.self_pass_active = False
        self.real_pass_active = False
        self.marl_self_pass_list = [False, False, False]
        self.marl_real_pass_list = [False, False, False]
        self.opp_possession_time = 0.0
        self.prev_poss_at_touch = None  # frozen at moment of touch, updated only on touch events
        self.last_action_kick = 0

        if getattr(self, 'phase', '') == 'A3':
            return [self._get_obs(agent_idx=i) for i in range(3)], {}
        else:
            return self._get_obs(agent_idx=0), {}

    def _reset_positions(self):
        size_class = '3v3' if getattr(self, 'phase', '') in ['A0.1', 'A3'] else '1v1'
        cands = [p for p in MAP_PRESETS if p[3] == size_class] or MAP_PRESETS
        preset = cands[int(self._rng.integers(0, len(cands)))]
        
        self.HH = float(preset[1])
        self.HW = float(preset[0])
        self.goal_y = float(preset[2])
        self.goal_center_y = 0.0

        # Map Randomization for A0
        if getattr(self, 'phase', '') == 'A0' and size_class == '1v1':
            scale = self._rng.uniform(0.8, 1.1)
            self.HW *= scale
            self.HH *= scale
            self.goal_y = self._rng.uniform(60.0, 70.0)
        
        # Determine opponent type
        if getattr(self, 'override_opponent_policy', None) is not None:
            self.opponent_types = ['Trained'] * 10
            self.opponent_type = 'Trained'
            self.opponent_policy = self.override_opponent_policy
        else:
            bot_pool = ['Random', 'Pazzo', 'Wanderer', 'Static']
            self.opponent_types = [self._rng.choice(bot_pool) for _ in range(10)]
            self.opponent_type = self.opponent_types[0]
            self.opponent_policy = None
            
        self.episode_type = 'opponent'

        # Positions
        self.agents = []
        num_per_team = 3 if getattr(self, 'phase', '') == 'A3' else 1
        
        # Red team
        for _ in range(num_per_team):
            x = float(self._rng.uniform(-self.HW + PLYR_R, 0 - PLYR_R))
            y = float(self._rng.uniform(-self.HH + PLYR_R, self.HH - PLYR_R))
            self.agents.append(Disc(x, y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
            
        # Blue team
        if self.n_agents > 1 or getattr(self, 'phase', '') == 'A3':
            for _ in range(num_per_team):
                x = float(self._rng.uniform(0 + PLYR_R, self.HW - PLYR_R))
                y = float(self._rng.uniform(-self.HH + PLYR_R, self.HH - PLYR_R))
                self.agents.append(Disc(x, y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))

        # Re-arrange based on team_id
        if self.team_id == 2:
            self.agents = self.agents[num_per_team:] + self.agents[:num_per_team]

        # Spawn ball
        if getattr(self, 'phase', '') == 'A3':
            self.ball = Disc(0.0, 0.0, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
        else:
            bx = float(self._rng.uniform(-self.HW + BALL_R, self.HW - BALL_R))
            by = float(self._rng.uniform(-self.HH + BALL_R, self.HH - BALL_R))
            self.ball = Disc(bx, by, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)

        # ── Special Training (>= 2M steps) ──
        if getattr(self, 'phase', '') == 'A3':
            return

        ts = getattr(self, 'total_timesteps_elapsed', 0)
        atk = self._attack_sign
        own_goal_x = -self.HW * atk
        opp_goal_x = self.HW * atk
        
        if ts >= 3_000_000:
            r = self._rng.random()
            if r < 0.2:
                # 20% ball right in front of own goal, agent in kick range, angle to own goal
                gap = max(0, self.goal_y - 10)
                y_pos = float(self._rng.uniform(-gap, gap))
                self.ball.x = float(own_goal_x + 20.0 * atk)
                self.ball.y = y_pos
                
                dx = own_goal_x - self.ball.x
                dy = 0.0
                dist = math.hypot(dx, dy)
                if dist > 0:
                    dx /= dist; dy /= dist
                else:
                    dx = atk; dy = 0.0
                    
                place_dist = PLYR_R + BALL_R + (KICK_RANGE * 0.5)
                self.agents[0].x = float(self.ball.x - dx * place_dist)
                self.agents[0].y = float(self.ball.y - dy * place_dist)
                
            elif r < 0.45:
                # 25% ball right at the sideline (top or bottom), agent nearby
                is_top = self._rng.random() < 0.5
                y_pos = float(-self.HH + BALL_R if is_top else self.HH - BALL_R)
                x_pos = float(self._rng.uniform(-self.HW + BALL_R + 50, self.HW - BALL_R - 50))
                
                self.ball.x = x_pos
                self.ball.y = y_pos
                
                # Place agent close to the ball
                agent_y_offset = (PLYR_R + BALL_R + 2.0) if is_top else -(PLYR_R + BALL_R + 2.0)
                self.agents[0].x = x_pos
                self.agents[0].y = y_pos + agent_y_offset
                
            elif r < 0.7:
                # 25% ball right at the endline (left or right), agent nearby
                is_left = self._rng.random() < 0.5
                x_pos = float(-self.HW + BALL_R if is_left else self.HW - BALL_R)
                
                # Must avoid spawning inside or directly in front of the goal
                is_top_half = self._rng.random() < 0.5
                if is_top_half:
                    y_pos = float(self._rng.uniform(-self.HH + BALL_R, -self.goal_y - BALL_R - 5.0))
                else:
                    y_pos = float(self._rng.uniform(self.goal_y + BALL_R + 5.0, self.HH - BALL_R))
                    
                self.ball.x = x_pos
                self.ball.y = y_pos
                
                # Place agent close to the ball
                agent_x_offset = (PLYR_R + BALL_R + 2.0) if is_left else -(PLYR_R + BALL_R + 2.0)
                self.agents[0].x = x_pos + agent_x_offset
                self.agents[0].y = y_pos
                
            else:
                # 30% random (do nothing, keep standard spawn)
                pass
            
        elif ts >= 2_000_000:
            r = self._rng.random()
            
            if r < 0.3:
                # 30% random (do nothing, keep standard spawn)
                pass
            elif r < 0.6:
                # 30% at opponent's HW sideline, but not inside [-goal_y, goal_y]
                y_cands = []
                if self.HH - PLYR_R > self.goal_y + PLYR_R:
                    y_cands.append((self.goal_y + PLYR_R, self.HH - PLYR_R))
                if -self.HH + PLYR_R < -self.goal_y - PLYR_R:
                    y_cands.append((-self.HH + PLYR_R, -self.goal_y - PLYR_R))
                
                if y_cands:
                    yrange = y_cands[self._rng.integers(0, len(y_cands))]
                    y_pos = float(self._rng.uniform(yrange[0], yrange[1]))
                    
                    self.ball.x = float(opp_goal_x - 15.0 * atk)
                    self.ball.y = y_pos
                    self.agents[0].x = float(self.ball.x - 30.0 * atk)
                    self.agents[0].y = y_pos
            elif r < 0.9:
                # 30% right in front of own goal
                gap = max(0, self.goal_y - 10)
                y_pos = float(self._rng.uniform(-gap, gap))
                self.ball.x = float(own_goal_x + 30.0 * atk)
                self.ball.y = y_pos
                touch_dist = PLYR_R + BALL_R + 0.1
                self.agents[0].x = float(self.ball.x + touch_dist * atk)
                self.agents[0].y = y_pos
            else:
                # 10% ball right in front of own goal, agent in kick range, angle to own goal
                gap = max(0, self.goal_y - 10)
                y_pos = float(self._rng.uniform(-gap, gap))
                self.ball.x = float(own_goal_x + 20.0 * atk)
                self.ball.y = y_pos
                
                # Agent placed opposite to own goal from ball (so kicking sends it to own goal)
                dx = own_goal_x - self.ball.x
                dy = 0.0
                dist = math.hypot(dx, dy)
                if dist > 0:
                    dx /= dist; dy /= dist
                else:
                    dx = atk; dy = 0.0
                    
                place_dist = PLYR_R + BALL_R + (KICK_RANGE * 0.5)
                self.agents[0].x = float(self.ball.x - dx * place_dist)
                self.agents[0].y = float(self.ball.y - dy * place_dist)

            # Prevent other agents from spawning on top of the special spawn
            for i in range(1, len(self.agents)):
                ag = self.agents[i]
                if math.hypot(ag.x - self.ball.x, ag.y - self.ball.y) < PLYR_R + BALL_R + 10 or \
                   math.hypot(ag.x - self.agents[0].x, ag.y - self.agents[0].y) < PLYR_R * 2 + 10:
                    safe_x, safe_y = self._safe_spawn()
                    ag.x, ag.y = safe_x, safe_y

    def _safe_spawn(self, max_tries: int = 80, x_min: float = None, x_max: float = None):
        """Return (x, y) not overlapping ball or already-placed agents.
        x_min/x_max optionally restrict the x range (for half-field spawning).
        """
        min_r_ball = PLYR_R + BALL_R + 8.0
        min_r_plyr = PLYR_R * 2 + 8.0
        HW, HH = self.HW, self.HH
        x_lo = x_min if x_min is not None else -HW * 0.85 + PLYR_R
        x_hi = x_max if x_max is not None else  HW * 0.85 - PLYR_R

        for _ in range(max_tries):
            x = float(self._rng.uniform(x_lo, x_hi))
            y = float(self._rng.uniform(-HH * 0.85 + PLYR_R, HH * 0.85 - PLYR_R))

            if math.hypot(x - self.ball.x, y - self.ball.y) < min_r_ball:
                continue

            ok = all(
                math.hypot(x - d.x, y - d.y) >= min_r_plyr
                for d in self.agents
            )
            if ok:
                return (x, y)

        # Fallback: centre of the requested x range, opposite side of field in y
        fx = (x_lo + x_hi) / 2.0
        fy = -HH * 0.5 if self.ball.y > 0 else HH * 0.5
        return (fx, fy)



    # ─── step ─────────────────────────────────────────────────────────────────
    def _predict_greedy_receiver(self):
        """Predicts which player will intercept the ball's current trajectory first."""
        bx, by = self.ball.x, self.ball.y
        bxs, bys = self.ball.xs, self.ball.ys
        
        # Simulate ball trajectory for up to 150 frames (2.5 seconds)
        for t in range(1, 151):
            bx += bxs
            by += bys
            bxs *= self.ball.damp
            bys *= self.ball.damp
            
            for ag in self.agents:
                dist = math.hypot(ag.x - bx, ag.y - by)
                # Player reach: radius + ball_radius + max_speed * t
                # max_speed is roughly 3.0
                if dist <= (ag.radius + self.ball.radius + 3.0 * t):
                    return ag
                    
        # If no one intercepts within 2.5s, return the closest agent to the ball's final predicted position
        closest_ag = None
        min_dist = float('inf')
        for ag in self.agents:
            dist = math.hypot(ag.x - bx, ag.y - by)
            if dist < min_dist:
                min_dist = dist
                closest_ag = ag
                
        return closest_ag

    def step(self, action):
        assert self.ball is not None, "Call reset() before step()"

        self.kick_exhausted = [False] * len(self.agents)

        agent_actions = []
        is_marl = getattr(self, 'phase', '') == 'A3'
        
        if is_marl:
            for act in action:
                dir_idx = int(act[0])
                kick    = int(act[1])
                dx, dy  = DIR_MAP[dir_idx]
                agent_actions.append((dx * self._flip, dy, kick))
            self.last_action_kick = int(action[0][1])
        else:
            dir_idx = int(action[0])
            kick    = int(action[1])
            self.last_action_kick = kick
            dx, dy  = DIR_MAP[dir_idx]
            agent_actions.append((dx * self._flip, dy, kick))
        
        # Determine opponent action
        if self.phase in ('A1', 'A1.2', 'A0.1', 'A3') and len(self.agents) > len(agent_actions):
            num_opps = len(self.agents) - len(agent_actions)
            num_agents = len(agent_actions)
            for opp_i in range(num_opps):
                ag = self.agents[num_agents + opp_i]
                opp_type = self.opponent_types[opp_i] if hasattr(self, 'opponent_types') else self.opponent_type
                
                if opp_type == 'Defender':
                    agent_actions.append(self._get_defender_action(ag))
                elif opp_type == 'Attacker':
                    agent_actions.append(self._get_attacker_action(ag))
                elif opp_type == 'Hybrid':
                    if self.last_touch == 'A':
                        self._hybrid_mode = 'defender'
                    elif self.last_touch == 'O':
                        self._hybrid_mode = 'follower'
                    if self._hybrid_mode == 'defender':
                        agent_actions.append(self._get_defender_action(ag))
                    else:
                        agent_actions.append(self._get_follower_action(ag))
                elif opp_type == 'Trained' and self.opponent_policy is not None:
                    opp_obs = self._get_obs_for_opponent(opp_i)
                    opp_action, _ = self.opponent_policy.predict(opp_obs, deterministic=False)
                    opp_dx, opp_dy = DIR_MAP[int(opp_action[0])]
                    agent_actions.append((opp_dx * -self._flip, opp_dy, int(opp_action[1])))
                elif opp_type == 'Human':
                    agent_actions.append(self.human_opponent_action)
                elif opp_type == 'Static':
                    agent_actions.append(self._get_static_action())
                elif opp_type == 'Random':
                    agent_actions.append(self._get_random_action())
                elif opp_type == 'Pazzo':
                    agent_actions.append(self._get_pazzo_action(ag, opp_i))
                elif opp_type == 'Wanderer':
                    agent_actions.append(self._get_wanderer_action(ag, opp_i))
                else:
                    agent_actions.append((0.0, 0.0, 0))

        goal_result = 0
        touch_events = []
        
        for _ in range(self.frame_skip):
            self.touches_this_tick = []
            result = self._tick(agent_actions)
            for touch in self.touches_this_tick:
                touch_events.append(touch)
                
            if result != 0:
                goal_result = result
                break

        # ── MARL Possession & Investment Sequence Logic ───────────────────────
        self.step_count += 1
        time_now = self.step_count * (self.frame_skip / 60.0)

        # prev_poss_0_25: frozen at moment of last touch. Read from self.prev_poss_at_touch.
        # Updated inside the touch loop below.
        prev_poss_0_25 = self.prev_poss_at_touch

        opp_id = 2 if self.team_id == 1 else 1

        turnover_penalty = 0.0
        for pid, tid in touch_events:
            # Capture prev_poss at the moment of touch.
            # prev = who had it before (last_touch_team), valid only within 0.25s.
            dt = time_now - self.last_touch_time
            if self.last_touch_team is not None and self.last_touch_team != tid and dt <= 0.25:
                # Previous possessor was a different team and within 0.25s window
                self.prev_poss_at_touch = self.last_touch_team
            else:
                # Same team (continuous hold) or ball was free too long → None
                self.prev_poss_at_touch = None
            prev_poss_0_25 = self.prev_poss_at_touch

            # Detect turnover: Ball was ours, now touched by opponent
            if self.last_touch_team == self.team_id and tid == opp_id:
                turnover_penalty -= 0.1

            self.last_touch = 'A' if pid == 0 else 'O'
            self.last_touch_team = tid
            self.last_touch_time = time_now

            # Reset dribble timer if different player touched
            if hasattr(self, '_last_touch_pid') and self._last_touch_pid != pid:
                self.dribble_start_time = time_now
            elif not hasattr(self, '_last_touch_pid'):
                self.dribble_start_time = time_now
            self._last_touch_pid = pid

            # Investment Sequence Tracking
            self.self_pass_active = False
            self.real_pass_active = False
            self.marl_self_pass_list = [False, False, False]
            self.marl_real_pass_list = [False, False, False]
            if tid == opp_id:
                pass
            else:
                # Teammate (or self) touched
                if pid in self.investment_sequence:
                    idx = self.investment_sequence.index(pid)
                    self.investment_sequence = self.investment_sequence[:idx+1]
                else:
                    self.investment_sequence.append(pid)

        if self.last_touch_team == opp_id:
            self.opp_possession_time += (self.frame_skip / 60.0)
            if self.opp_possession_time >= 2.0:
                self.investment_sequence.clear()
        else:
            self.opp_possession_time = 0.0

        # ── Reward ────────────────────────────────────────────────────────────
        reward = 0.0
        terminated = False
        truncated = False
        
        atk = self._attack_sign
        goal_x = self.HW * atk

        # Compute Zones for X-axis and Delta Dist
        bx, by = self.ball.x, self.ball.y
        px, py = self._prev_ball_x, self._prev_ball_y
        
        adv_x = bx * atk
        zone_width = self.goal_y * 2.0
        
        if adv_x <= -self.HW + zone_width:
            own_goal_x = -self.HW * atk
            cur_dist = math.hypot(bx - own_goal_x, by)
            prev_dist = math.hypot(px - own_goal_x, py)
            delta_dist_to_goal = cur_dist - prev_dist
        elif adv_x >= self.HW - zone_width:
            opp_goal_x = self.HW * atk
            cur_dist = math.hypot(bx - opp_goal_x, by)
            prev_dist = math.hypot(px - opp_goal_x, py)
            delta_dist_to_goal = prev_dist - cur_dist
        else:
            delta_dist_to_goal = (bx - px) * atk

        prev_poss_our_side = (prev_poss_0_25 == self.team_id)
        has_possession_reward = (self.last_touch_team == self.team_id) or prev_poss_our_side
        has_possession_penalty = True
        
        is_opp_pass_back = (prev_poss_0_25 == opp_id and self.last_touch_team == opp_id)
        has_teammates = len(self.agents) > 1
        is_pass_back = has_teammates and (prev_poss_our_side and self.last_touch_team == self.team_id)
        
        dribble_duration = time_now - getattr(self, 'dribble_start_time', time_now)
        is_dribbling = (dribble_duration > 0.0)
        
        # Frozen Reward (Events)
        frozen_reward = 0.0
        if goal_result == 2:
            frozen_reward += 10.0
        elif goal_result == 1:
            frozen_reward -= 15.0
            
        ts = getattr(self, 'total_timesteps_elapsed', 0)
        
        if getattr(self, 'phase', '') == 'A3':
            rew_list = [0.0] * 3
            
            # Anti-self pass detection using Greedy Prediction
            predicted_receiver = None
            has_valid_kick = False
            for i in range(3):
                if agent_actions[i][2] == 1 and getattr(self, '_last_touch_pid', None) == i:
                    has_valid_kick = True
                
            if has_valid_kick:
                predicted_receiver = self._predict_greedy_receiver()
                for i in range(3):
                    if agent_actions[i][2] == 1 and getattr(self, '_last_touch_pid', None) == i and predicted_receiver is not None:
                        if predicted_receiver == self.agents[i]:
                            self.marl_self_pass_list[i] = True
                            self.marl_real_pass_list[i] = False
                        elif predicted_receiver in self.agents[:3]:
                            self.marl_self_pass_list[i] = False
                            self.marl_real_pass_list[i] = True
                
            # Global goal logic
            goal_scored = False
            if goal_result == 2:
                self.scores[self.team_id - 1] += 1
                base_reward = 20.0
                bonus_pool = 3.0
                for i in range(3): rew_list[i] += base_reward
                
                # Assister bonus
                for i in range(3):
                    if i in self.investment_sequence and self.investment_sequence[-1] != i:
                        seq_idx = self.investment_sequence.index(i)
                        passes_away = len(self.investment_sequence) - 1 - seq_idx
                        invest_share = 0.3 * (0.5 ** (passes_away - 1))
                        rew_list[i] += bonus_pool * invest_share
                        
                goal_scored = True
                terminated = True
                
            elif goal_result == 1:
                self.scores[opp_id - 1] += 1
                for i in range(3): rew_list[i] -= 20.0
                goal_scored = True
                terminated = True

            if not terminated and self.step_count >= self.max_steps:
                truncated = True
                
            # Per-agent reward evaluation
            self_pass_list = [False, False, False]
            for i in range(3):
                # We skip kick penalty per user feedback
                
                # Check pass
                self_pass = self.marl_self_pass_list[i]
                real_pass = self.marl_real_pass_list[i]
                        
                # Invest share
                min_share = 0.3 * (0.5 ** 3) # 3 team members
                invest_share = min_share
                in_sequence = False
                if self.investment_sequence:
                    if self.investment_sequence[-1] == i:
                        invest_share = 1.0
                        in_sequence = True
                    elif i in self.investment_sequence:
                        seq_idx = self.investment_sequence.index(i)
                        passes_away = len(self.investment_sequence) - 1 - seq_idx
                        invest_share = 0.3 * (0.5 ** (passes_away - 1))
                        in_sequence = True
                        
                # Advance / Backward
                ADVANCE_REWARD = 0.003
                BACKWARD_PENALTY = 0.003
                
                if delta_dist_to_goal > 0:
                    if has_possession_reward and not is_opp_pass_back:
                        penalty_mult = 0.333 if is_pass_back else 1.0
                        mult = 1.0
                        if real_pass: mult = 1.0
                        elif self_pass: mult = penalty_mult
                        elif is_dribbling and agent_actions[i][2] == 0: mult = penalty_mult
                        rew_list[i] += ADVANCE_REWARD * delta_dist_to_goal * mult * invest_share
                elif delta_dist_to_goal < 0:
                    if has_possession_penalty:
                        mult = 1.0
                        if is_pass_back: mult = 0.333
                        
                        penalty_share = 1.0
                        if self.last_touch_team != opp_id:
                            penalty_share = invest_share if in_sequence else 1.0
                            
                        rew_list[i] -= BACKWARD_PENALTY * abs(delta_dist_to_goal) * mult * penalty_share
                        
                # Turnover Penalty
                if turnover_penalty < 0.0:
                    if in_sequence:
                        rew_list[i] += turnover_penalty * invest_share
                    else:
                        rew_list[i] += turnover_penalty * 1.0
                        
                # Minor dense reward for ball speed (only early in training)
                if ts < 3_000_000:
                    ball_speed = math.hypot(self.ball.xs, self.ball.ys)
                    if ball_speed > 0.3:
                        rew_list[i] += 0.001
            
            # Update history
            self._prev_dist_to_ball = math.hypot(self.agents[0].x - self.ball.x, self.agents[0].y - self.ball.y)
            self._prev_ball_dist_to_goal = _dist_to_goal_segment(self.ball.x, self.ball.y, goal_x, self.goal_y, -self.goal_y)
            self._prev_ball_x       = self.ball.x
            self._prev_ball_y       = self.ball.y
            self._prev_ball_speed   = math.hypot(self.ball.xs, self.ball.ys)

            info = {
                "marl/sequence_len": len(self.investment_sequence),
                "marl/dribble_duration": getattr(self, 'dribble_duration', 0.0),
                "marl/opp_pos_time": float(self.opp_possession_time),
                "marl/self_pass": list(self.marl_self_pass_list),
            }
            obs_list = [self._get_obs(agent_idx=i) for i in range(3)]
            return obs_list, rew_list, terminated, truncated, [info]*3

        else:
            # ── ORIGINAL REWARD BLOCK FOR A0/A1 ──
            # Anti-self pass detection on KICK using Greedy Prediction
            kick = getattr(self, 'last_action_kick', 0)
            if kick and self.last_touch_team == self.team_id:
                receiver = self._predict_greedy_receiver()
                if receiver is self.agents[0]:
                    self.self_pass_active = True
                    self.real_pass_active = False
                else:
                    self.self_pass_active = False
                    self.real_pass_active = True
            self.real_pass_active = False

            reward = 0.0
            dist = math.hypot(self.ball.x - self.agents[0].x, self.ball.y - self.agents[0].y)
            in_range = dist - self.agents[0].radius - self.ball.radius < KICK_RANGE
            if self.last_action_kick == 1 and not in_range:
                reward -= 0.001

            # Determine invest_share for the current agent
            num_team_members = len(self.agents) // 2 if len(self.agents) > 1 else 1
            min_share = 0.3 * (0.5 ** num_team_members)
            invest_share = min_share
            in_sequence = False
            if self.investment_sequence:
                if self.investment_sequence[-1] == 0:
                    invest_share = 1.0
                    in_sequence = True
                elif 0 in self.investment_sequence:
                    seq_idx = self.investment_sequence.index(0)
                    passes_away = len(self.investment_sequence) - 1 - seq_idx
                    invest_share = 0.3 * (0.5 ** (passes_away - 1))
                    in_sequence = True

            ADVANCE_REWARD = 0.003
            BACKWARD_PENALTY = 0.003
            
            if delta_dist_to_goal > 0: # Ball advanced (good)
                if has_possession_reward and not is_opp_pass_back:
                    penalty_mult = 0.333 if is_pass_back else 1.0
                    mult = 1.0
                    if self.real_pass_active:
                        mult = 1.0
                    elif self.self_pass_active:
                        mult = penalty_mult
                    elif is_dribbling and not kick:
                        mult = penalty_mult
                    reward += ADVANCE_REWARD * delta_dist_to_goal * mult * invest_share
                    
            elif delta_dist_to_goal < 0: # Ball moved back (bad)
                if has_possession_penalty:
                    mult = 1.0
                    if is_pass_back:
                        mult = 0.333
                    if self.last_touch_team == opp_id:
                        penalty_share = 1.0
                    else:
                        penalty_share = invest_share if in_sequence else 1.0
                    reward -= BACKWARD_PENALTY * abs(delta_dist_to_goal) * mult * penalty_share

            if turnover_penalty < 0.0:
                if in_sequence:
                    reward += turnover_penalty * invest_share
                else:
                    reward += turnover_penalty * 1.0

            goal_scored = False
            if goal_result == 2:
                self.scores[self.team_id - 1] += 1
                base_reward = 10.0
                bonus_pool = 3.0
                reward += base_reward
                if self.investment_sequence and self.investment_sequence[-1] != 0 and 0 in self.investment_sequence:
                    seq_idx = self.investment_sequence.index(0)
                    passes_away = len(self.investment_sequence) - 1 - seq_idx
                    invest_share = 0.3 * (0.5 ** (passes_away - 1))
                    reward += bonus_pool * invest_share
                goal_scored = True
                terminated = True
                
            elif goal_result == 1:
                self.scores[opp_id - 1] += 1
                reward -= 15.0
                goal_scored = True
                terminated = True

            if not terminated and self.step_count >= self.max_steps:
                truncated = True

            self._prev_dist_to_ball = math.hypot(self.agents[0].x - self.ball.x, self.agents[0].y - self.ball.y)
            self._prev_ball_dist_to_goal = _dist_to_goal_segment(self.ball.x, self.ball.y, goal_x, self.goal_y, -self.goal_y)
            self._prev_ball_x       = self.ball.x
            self._prev_ball_y       = self.ball.y
            self._prev_ball_speed   = math.hypot(self.ball.xs, self.ball.ys)

            if ts < 3_000_000:
                ball_speed = math.hypot(self.ball.xs, self.ball.ys)
                if ball_speed > 0.3:
                    frozen_reward += 0.001
                if goal_result == 2:
                    frozen_reward += 0.001 * (self.max_steps - self.step_count)
            else:
                if goal_result == 2:
                    frozen_reward += 0.02 * (self.max_steps - self.step_count)
                    
            if ts >= 3_000_000:
                reward = frozen_reward
            else:
                reward += frozen_reward
                
            info = {
                "marl/sequence_len": len(self.investment_sequence),
                "marl/dribble_duration": getattr(self, 'dribble_duration', 0.0),
                "marl/self_pass": int(getattr(self, 'self_pass_active', False)),
                "marl/opp_pos_time": float(self.opp_possession_time),
            }
            return self._get_obs(agent_idx=0), float(reward), terminated, truncated, info


    # ─── Physics ──────────────────────────────────────────────────────────────
    def _tick(self, agent_actions) -> int:
        """
        One physics tick. Takes a list of (dx, dy, kick) for each agent.
        Returns 0=normal | 1=left-goal | 2=right-goal.
        """
        ball   = self.ball
        agents = self.agents
        HW, HH = self.HW, self.HH

        action_order = list(range(len(agents)))
        self._rng.shuffle(action_order)
        
        if not hasattr(self, 'kick_exhausted') or len(self.kick_exhausted) != len(agents):
            self.kick_exhausted = [False] * len(agents)

        masked_actions = []
        for i in range(len(agents)):
            dx, dy, kick = agent_actions[i]
            
            # Reset exhausted state if they release the kick button
            if kick == 0:
                self.kick_exhausted[i] = False
                masked_actions.append((dx, dy, 0))
            else:
                # If they press kick, only accept if in range AND not exhausted
                dist = math.hypot(ball.x - agents[i].x, ball.y - agents[i].y)
                in_range = dist - agents[i].radius - ball.radius < KICK_RANGE
                
                if in_range and not self.kick_exhausted[i]:
                    masked_actions.append((dx, dy, 1))
                else:
                    masked_actions.append((dx, dy, 0))

        # 1. Kick (before movement) — random order
        for i in action_order:
            ag = agents[i]
            dx, dy, kick = masked_actions[i]
            if kick:
                dx_b = ball.x - ag.x
                dy_b = ball.y - ag.y
                dist = math.hypot(dx_b, dy_b)
                if dist > 0 and dist - ag.radius - ball.radius < KICK_RANGE:
                    nx, ny = dx_b / dist, dy_b / dist
                    ball.xs += nx * KICK_STR
                    ball.ys += ny * KICK_STR
                    self.kick_exhausted[i] = True  # Consumed!
                    
                    num_per_team = len(agents) // 2 if len(agents) > 1 else 1
                    team = self.team_id if i < num_per_team else (2 if self.team_id == 1 else 1)
                    self.touches_this_tick.append((i, team))

        # 2. Acceleration — random order
        for i in action_order:
            ag = agents[i]
            dx, dy, _ = masked_actions[i]
            intent_kick = agent_actions[i][2]
            ln = math.hypot(dx, dy)
            acc = PLYR_KICK_ACC if intent_kick else PLYR_ACC
            if ln > 0:
                ndx, ndy = dx / ln, dy / ln
                ag.xs += ndx * acc
                ag.ys += ndy * acc

        # 3. Move all
        ball.x += ball.xs; ball.y += ball.ys
        for ag in agents:
            ag.x += ag.xs; ag.y += ag.ys

        # 4. Disc-Disc collisions (Players and Ball)
        shuffled_agents = list(agents)
        self._rng.shuffle(shuffled_agents)
        all_discs = [ball] + shuffled_agents
        n = len(all_discs)
        
        for i in range(n):
            for j in range(i + 1, n):
                collided = _resolve_dd(all_discs[i], all_discs[j])
                if collided:
                    if (i == 0 and j > 0) or (j == 0 and i > 0):
                        # ball is index 0. The other is max(i, j)
                        ag_disc = all_discs[max(i, j)]
                        # Find index in original self.agents
                        try:
                            orig_idx = self.agents.index(ag_disc)
                            num_per_team = len(self.agents) // 2 if len(self.agents) > 1 else 1
                            team = self.team_id if orig_idx < num_per_team else (2 if self.team_id == 1 else 1)
                            self.touches_this_tick.append((orig_idx, team))
                        except ValueError:
                            pass
                            
        # 4b. Goal Poles Collisions
        poles = [
            Disc( HW, self.goal_center_y + self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0), # Right Top
            Disc( HW, self.goal_center_y - self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0), # Right Bottom
            Disc(-HW, self.goal_center_y + self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0), # Left Top
            Disc(-HW, self.goal_center_y - self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0), # Left Bottom
        ]
        
        # Check pole collisions against ball and players
        for pole in poles:
            for ag in agents:
                _resolve_dd(ag, pole)
            _resolve_dd(ball, pole)

        # 5. Wall collisions — Ball
        if ball.y - ball.radius < -HH:
            ball.y = -HH + ball.radius
            ball.ys = -ball.ys * ball.bcoef
        if ball.y + ball.radius > HH:
            ball.y = HH - ball.radius
            ball.ys = -ball.ys * ball.bcoef

        atk = self._attack_sign
        if ball.x - ball.radius < -HW:
            # Within goal posts?
            if self.goal_center_y - self.goal_y < ball.y < self.goal_center_y + self.goal_y:
                return 1 if atk == 1 else 2
            
            # Not in goal, bounce off back wall line
            ball.x = -HW + ball.radius
            ball.xs = -ball.xs * ball.bcoef
            
        elif ball.x + ball.radius > HW:
            # Within goal posts?
            if self.goal_center_y - self.goal_y < ball.y < self.goal_center_y + self.goal_y:
                return 2 if atk == 1 else 1
            
            # Not in goal, bounce off back wall line
            ball.x = HW - ball.radius
            ball.xs = -ball.xs * ball.bcoef

        # Wall collisions — Players: OUTER_PAD limit
        # Players cannot exceed the stadium boundaries. No bounciness.
        max_y = HH + OUTER_PAD
        max_x = HW + GOAL_DEPTH
        for ag in agents:
            if ag.x - ag.radius < -max_x:
                ag.x = -max_x + ag.radius
                if ag.xs < 0: ag.xs = 0.0
            elif ag.x + ag.radius > max_x:
                ag.x = max_x - ag.radius
                if ag.xs > 0: ag.xs = 0.0
                
            if ag.y - ag.radius < -max_y:
                ag.y = -max_y + ag.radius
                if ag.ys < 0: ag.ys = 0.0
            elif ag.y + ag.radius > max_y:
                ag.y = max_y - ag.radius
                if ag.ys > 0: ag.ys = 0.0

        # 6. Damping
        ball.xs *= ball.damp; ball.ys *= ball.damp
        for ag in agents:
            ag.xs *= ag.damp; ag.ys *= ag.damp

        return 0

    # ─── Observation ──────────────────────────────────────────────────────────
    def _get_obs(self, agent_idx=0) -> np.ndarray:
        ball  = self.ball
        agent = self.agents[agent_idx]
        obs   = np.zeros(OBS_DIM, dtype=np.float32)

        bx, by   = ball.x,   ball.y
        bxs, bys = ball.xs,  ball.ys
        mx, my   = agent.x,  agent.y
        mxs, mys = agent.xs, agent.ys
        flip = self._flip

        surf_dist_raw = math.hypot(mx - bx, my - by) - PLYR_R - BALL_R
        surf_dist = max(0.0, surf_dist_raw)
        dist_diff = surf_dist_raw - KICK_RANGE

        i = 0

        # Section 1 — Field constants (4)
        obs[i] = self.goal_y / NORM;                    i += 1
        obs[i] = self.HH / NORM;                        i += 1
        obs[i] = self.HW / NORM;                        i += 1
        obs[i] = 0.0;    i += 1  # 0=RED, 1=BLUE

        # Section 2 — Agent ↔ Ball (7)
        obs[i] = flip * (bx - mx) / NORM;     i += 1
        obs[i] = (by - my) / NORM;            i += 1
        obs[i] = surf_dist / DIAG;            i += 1
        obs[i] = dist_diff / DIAG;            i += 1
        
        my_dist_to_ball = math.hypot(mx - bx, my - by)
        
        # Determine if agent is the closest player on the entire field
        is_nearest_field = 1.0
        for ag in self.agents:
            if ag != agent and math.hypot(ag.x - bx, ag.y - by) < my_dist_to_ball:
                is_nearest_field = 0.0
                break
        obs[i] = is_nearest_field; i += 1
        
        # Determine if agent is the closest on their team, and their rank on team
        my_team_agents = [self.agents[0]]
        # Because training/env.py places main agent at 0, teammates at 2..N_TM+1 if N_TM>0.
        # Wait, how are agents ordered? [A0_RED, A0_BLUE, A1_RED, A1_BLUE...] 
        # Typically agents list is structured based on team. Let's just calculate distances for all team agents.
        team_dists = []
        for j, ag in enumerate(self.agents):
            # Same team if index % 2 == 0 (assuming team 1) vs index % 2 == 1 (team 2),
            # or based on how env adds them. Let's safely check their intended team.
            # In self-play, agents[0], agents[2], ... are Red, agents[1], agents[3]... are Blue
            # Actually, `ag` doesn't have a team attribute directly here. We can just use the index parity.
            if j % 2 == 0:
                team_dists.append(math.hypot(ag.x - bx, ag.y - by))
                
        team_dists.sort()
        is_nearest_team = 1.0 if (len(team_dists) > 0 and team_dists[0] >= my_dist_to_ball - 0.001) else 0.0
        obs[i] = is_nearest_team; i += 1
        
        # Rank on team (0 = closest, 1 = second, etc.)
        rank = 0.0
        for d in team_dists:
            if d < my_dist_to_ball - 0.001:
                rank += 1.0
        obs[i] = rank; i += 1

        # Section 2b — Ball ↔ Goals (8)
        top_post_y = self.goal_center_y - self.goal_y
        bot_post_y = self.goal_center_y + self.goal_y
        
        # Opponent Top Post
        obs[i] = (self.HW - flip * bx) / NORM;       i += 1
        obs[i] = (top_post_y - by) / NORM;           i += 1
        
        # Opponent Bottom Post
        obs[i] = (self.HW - flip * bx) / NORM;       i += 1
        obs[i] = (bot_post_y - by) / NORM;           i += 1

        # Own Top Post
        obs[i] = (-self.HW - flip * bx) / NORM;      i += 1
        obs[i] = (top_post_y - by) / NORM;           i += 1
        
        # Own Bottom Post
        obs[i] = (-self.HW - flip * bx) / NORM;      i += 1
        obs[i] = (bot_post_y - by) / NORM;           i += 1

        # Section 3 — Dynamic state (11)
        obs[i] = flip * bx / NORM;            i += 1
        obs[i] = (by - self.goal_center_y) / NORM;  i += 1
        obs[i] = flip * bxs / MAX_SPEED;      i += 1
        obs[i] = bys / MAX_SPEED;             i += 1
        obs[i] = flip * mx / NORM;            i += 1
        obs[i] = (my - self.goal_center_y) / NORM;  i += 1
        obs[i] = flip * mxs / MAX_SPEED;      i += 1
        obs[i] = mys / MAX_SPEED;             i += 1
        obs[i] = math.hypot(mxs, mys) / MAX_SPEED; i += 1
        obs[i] = flip * (mxs - bxs) / MAX_SPEED;   i += 1
        obs[i] = (mys - bys) / MAX_SPEED;          i += 1

        # Section 4 — Game state (2)
        obs[i] = max(0.0, 1.0 - self.step_count / self.max_steps); i += 1
        obs[i] = self.max_steps / MAX_STEPS_ALL_MODES;  i += 1

        # ── MARL Additions ──────────────────────────────────────────────────
        opp_id = 2 if self.team_id == 1 else 1

        # Possession Current (3 dims: None, Team, Opp)
        if self.last_touch_team is None:
            obs[i:i+3] = [1, 0, 0]
        elif self.last_touch_team == self.team_id:
            obs[i:i+3] = [0, 1, 0]
        else:
            obs[i:i+3] = [0, 0, 1]
        i += 3
        
        # Previous possession among 0.25s (3 dims: None, Team, Opp)
        # None = continuous hold OR ball was free >0.25s before last touch
        # Frozen at moment of last touch, not a rolling window
        _prev_poss_obs = self.prev_poss_at_touch
        if _prev_poss_obs is None:
            obs[i:i+3] = [1, 0, 0]
        elif _prev_poss_obs == self.team_id:
            obs[i:i+3] = [0, 1, 0]
        else:
            obs[i:i+3] = [0, 0, 1]
        i += 3
        
        # Opponent possession timer (1 dim)
        obs[i] = min(1.0, self.opp_possession_time / 2.0); i += 1

        # Investment sequence share for main agent (1 dim)
        num_team_members = len(self.agents) // 2 if len(self.agents) > 1 else 1
        min_share = 0.3 * (0.5 ** num_team_members)
        agent_share = min_share
        if self.investment_sequence:
            if self.investment_sequence[-1] == 0:
                agent_share = 1.0  # Holder
            elif 0 in self.investment_sequence:
                seq_idx = self.investment_sequence.index(0)
                # N is the number of passes away from the holder.
                # sequence = [0, 1] -> holder is 1. seq_idx = 0. length = 2.
                # passes away = len - 1 - seq_idx = 2 - 1 - 0 = 1
                passes_away = len(self.investment_sequence) - 1 - seq_idx
                agent_share = 0.3 * (0.5 ** (passes_away - 1))
        obs[i] = agent_share; i += 1

        def _get_tangent_vectors(px, py):
            dx = px - bx
            dy = py - by
            d2 = dx*dx + dy*dy
            R = PLYR_R + BALL_R
            if d2 <= R*R:
                return 0.0, 0.0, 0.0, 0.0
            d = math.sqrt(d2)
            L = math.sqrt(d2 - R*R)
            
            # Goc quet bat dau tu UP (-y), theo chieu kim dong ho
            theta = math.atan2(dx, -dy)
            alpha = math.asin(R / d)
            
            # 1. Theo chieu kim dong ho truoc (+ alpha)
            angle_cw = theta + alpha
            v1_x = L * math.sin(angle_cw)
            v1_y = -L * math.cos(angle_cw)
            
            # 2. Nguoc chieu kim dong ho sau (- alpha)
            angle_ccw = theta - alpha
            v2_x = L * math.sin(angle_ccw)
            v2_y = -L * math.cos(angle_ccw)
            
            return flip * v1_x / NORM, v1_y / NORM, flip * v2_x / NORM, v2_y / NORM

        # Main agent tangent vectors (4 dims)
        t_mx1, t_my1, t_mx2, t_my2 = _get_tangent_vectors(mx, my)
        obs[i:i+4] = [t_mx1, t_my1, t_mx2, t_my2]
        i += 4

        num_per_team = 3 if getattr(self, 'phase', '') == 'A3' else 1
        
        # Teammates
        tm_count = 0
        for j in range(num_per_team):
            if j == agent_idx: continue
            if tm_count >= N_TM: break
            
            tm = self.agents[j]
            idx = i + tm_count * 14
            
            obs[idx] = flip * tm.x / NORM;                 idx += 1
            obs[idx] = tm.y / NORM;                        idx += 1
            obs[idx] = flip * tm.xs / MAX_SPEED;           idx += 1
            obs[idx] = tm.ys / MAX_SPEED;                  idx += 1
            
            obs[idx] = flip * (tm.x - mx) / NORM;          idx += 1
            obs[idx] = (tm.y - my) / NORM;                 idx += 1
            
            obs[idx] = flip * (bx - tm.x) / NORM;          idx += 1
            obs[idx] = (by - tm.y) / NORM;                 idx += 1
            
            tm_surf_dist = max(0.0, math.hypot(tm.x - bx, tm.y - by) - PLYR_R - BALL_R)
            obs[idx] = tm_surf_dist / DIAG;                idx += 1
            
            t_tx1, t_ty1, t_tx2, t_ty2 = _get_tangent_vectors(tm.x, tm.y)
            obs[idx:idx+4] = [t_tx1, t_ty1, t_tx2, t_ty2]; idx += 4

            share = 0.0
            if self.investment_sequence:
                tm_idx = self.agents.index(tm)
                if self.investment_sequence[-1] == tm_idx:
                    share = 1.0
                elif tm_idx in self.investment_sequence:
                    seq_idx = self.investment_sequence.index(tm_idx)
                    passes_away = len(self.investment_sequence) - 1 - seq_idx
                    share = 0.3 * (0.5 ** (passes_away - 1))
            
            obs[idx] = share; idx += 1
            tm_count += 1
            
        i += N_TM * 14

        # Opponents
        opp_count = 0
        for j in range(num_per_team, len(self.agents)):
            if opp_count >= N_OPP: break
            
            opp = self.agents[j]
            idx = i + opp_count * 13
            
            obs[idx] = flip * opp.x / NORM;                 idx += 1
            obs[idx] = opp.y / NORM;                        idx += 1
            obs[idx] = flip * opp.xs / MAX_SPEED;           idx += 1
            obs[idx] = opp.ys / MAX_SPEED;                  idx += 1
            
            obs[idx] = flip * (opp.x - mx) / NORM;          idx += 1
            obs[idx] = (opp.y - my) / NORM;                 idx += 1
            
            obs[idx] = flip * (bx - opp.x) / NORM;          idx += 1
            obs[idx] = (by - opp.y) / NORM;                 idx += 1
            
            opp_surf_dist = max(0.0, math.hypot(opp.x - bx, opp.y - by) - PLYR_R - BALL_R)
            obs[idx] = opp_surf_dist / DIAG;                idx += 1
            
            t_ox1, t_oy1, t_ox2, t_oy2 = _get_tangent_vectors(opp.x, opp.y)
            obs[idx:idx+4] = [t_ox1, t_oy1, t_ox2, t_oy2]; idx += 4
            
            opp_count += 1

        i += N_OPP * 13

        return obs


    def _get_obs_for_opponent(self, opp_idx=0) -> np.ndarray:
        # Swap team context and return obs to opponent (for self play evaluation inside step)
        original_team = self.team_id
        # Swap explicitly
        old_flip = self._flip
        old_attack = self._attack_sign
        
        self.team_id = 2 if self.team_id == 1 else 1
        self._flip = 1.0 if self.team_id == 1 else -1.0
        self._attack_sign = 1 if self.team_id == 1 else -1
        
        # We also need to swap agents for the perspective of `_get_obs`
        num_per_team = 3 if getattr(self, 'phase', '') == 'A3' else 1
        team1 = self.agents[:num_per_team]
        team2 = self.agents[num_per_team:num_per_team*2]
        self.agents = team2 + team1 + self.agents[num_per_team*2:]
        
        obs_opp = self._get_obs(agent_idx=opp_idx)
        
        # Revert
        self.agents = team1 + team2 + self.agents[num_per_team*2:]
        self.team_id = original_team
        self._flip = old_flip
        self._attack_sign = old_attack
        return obs_opp

    def _get_follower_action(self, ag):
        """Chase ball and kick toward opponent goal.
        Anti-own-goal: simulate ball trajectory after kick (ray-goal intersection).
        - If kick would enter own goal → suppress + reposition laterally.
        - Otherwise → kick freely regardless of distance.
        """
        b  = self.ball
        b  = self.ball

        own_goal_x   = self.HW * self._attack_sign
        own_goal_top = self.goal_center_y + self.goal_y
        own_goal_bot = self.goal_center_y - self.goal_y

        dist_to_ball = math.hypot(b.x - ag.x, b.y - ag.y)
        in_range  = dist_to_ball - ag.radius - b.radius < KICK_RANGE

        # ── Ray-test: would this kick enter own goal? ──────────────────────────
        would_own_goal = False
        if in_range:
            kick_dx = b.x - ag.x
            kick_dy = b.y - ag.y
            k_dist  = math.hypot(kick_dx, kick_dy)
            if k_dist > 0:
                nx, ny = kick_dx / k_dist, kick_dy / k_dist
                # Resulting ball velocity after kick impulse
                vx = b.xs + nx * KICK_STR
                vy = b.ys + ny * KICK_STR
                # Time for ball to reach own goal x-line
                if abs(vx) > 0.01:
                    t = (own_goal_x - b.x) / vx
                    if t > 0:   # ball moving toward that x-line
                        y_at_goal = b.y + t * vy
                        if own_goal_bot < y_at_goal < own_goal_top:
                            would_own_goal = True

        # ── Action decision ───────────────────────────────────────────────────
        if would_own_goal:
            # Move laterally past the goal post to get a safe angle
            offset_y = self.goal_y + PLYR_R + 20
            target_y = b.y + (offset_y if ag.y <= b.y else -offset_y)
            target_x = b.x
            kick = 0
        else:
            target_x = b.x
            target_y = b.y
            kick = 1 if in_range else 0

        dx, dy = target_x - ag.x, target_y - ag.y
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            return (0, 0, 0)

        dx, dy = dx / dist, dy / dist
        best_dir_idx = 0
        best_dot = -2.0
        for i, (mx, my) in enumerate(DIR_MAP):
            dot = dx * mx + dy * my
            if dot > best_dot:
                best_dot = dot
                best_dir_idx = i

        out_dx, out_dy = DIR_MAP[best_dir_idx]
        return (out_dx, out_dy, kick)

    def _get_attacker_action(self, ag):
        """Aggressively tries to score into agent's goal.
        Positions behind the ball (own-goal side) so kicks go toward agent's goal.
        Has anti-own-goal ray test: never kicks into own goal.
        """
        b  = self.ball
        b  = self.ball

        own_goal_x   = self.HW * self._attack_sign   # follower's own goal
        own_goal_top = self.goal_center_y + self.goal_y
        own_goal_bot = self.goal_center_y - self.goal_y

        dist_to_ball = math.hypot(b.x - ag.x, b.y - ag.y)
        in_range = dist_to_ball - ag.radius - b.radius < KICK_RANGE

        # "Good side" = attacker is between own goal and ball (kick pushes ball away from own goal)
        # own_goal and ag should be on same side of ball in x
        on_good_side = (ag.x - b.x) * (own_goal_x - b.x) > 0

        # ── Anti-own-goal ray test ─────────────────────────────────────────────
        would_own_goal = False
        if in_range:
            kick_dx = b.x - ag.x
            kick_dy = b.y - ag.y
            k_dist  = math.hypot(kick_dx, kick_dy)
            if k_dist > 0:
                nx, ny = kick_dx / k_dist, kick_dy / k_dist
                vx = b.xs + nx * KICK_STR
                vy = b.ys + ny * KICK_STR
                if abs(vx) > 0.01:
                    t = (own_goal_x - b.x) / vx
                    if t > 0:
                        y_at = b.y + t * vy
                        if own_goal_bot < y_at < own_goal_top:
                            would_own_goal = True

        # ── Action decision ────────────────────────────────────────────────────
        if in_range and would_own_goal:
            # Lateral reposition to clear own goal line
            offset_y = self.goal_y + PLYR_R + 20
            target_x = b.x
            target_y = b.y + (offset_y if ag.y <= b.y else -offset_y)
            kick = 0
        elif in_range and not on_good_side:
            # In range but wrong side — go around ball without kicking
            offset_y = PLYR_R + BALL_R + 20
            target_x = b.x
            target_y = b.y + (offset_y if ag.y <= b.y else -offset_y)
            kick = 0
        elif in_range:
            # Good position — kick!
            target_x = b.x
            target_y = b.y
            kick = 1
        elif on_good_side:
            # Far but on correct side — approach ball directly
            target_x = b.x
            target_y = b.y
            kick = 0
        else:
            # Far and wrong side — navigate to behind the ball
            # Aim for a point on the own-goal side of ball
            behind_offset = (PLYR_R + BALL_R + 20) * self._attack_sign
            target_x = b.x + behind_offset   # land on own-goal side of ball
            target_y = b.y
            kick = 0

        dx, dy = target_x - ag.x, target_y - ag.y
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            return (0, 0, 0)

        dx, dy = dx / dist, dy / dist
        best_dir_idx = 0
        best_dot = -2.0
        for i, (mx, my) in enumerate(DIR_MAP):
            dot = dx * mx + dy * my
            if dot > best_dot:
                best_dot = dot
                best_dir_idx = i

        out_dx, out_dy = DIR_MAP[best_dir_idx]
        return (out_dx, out_dy, kick)


    def _get_static_action(self):
        """StaticModel: Always stays still"""
        return (0.0, 0.0, 0)

    def _get_random_action(self):
        """RandomModel: Randomly moves and kicks"""
        dir_idx = int(self._rng.integers(0, 9))
        kick = int(self._rng.integers(0, 2))
        dx, dy = DIR_MAP[dir_idx]
        return (dx, dy, kick)

    def _get_pazzo_action(self, ag, opp_i):
        """PazzoModel: Moves to a random point, changes every rand[100,150] steps. Jitters if ball is near center."""
        b = self.ball

        if not hasattr(self, '_pazzo_steps'):
            self._pazzo_steps = {}
            self._pazzo_target = {}
            self._pazzo_interval = {}
            
        if opp_i not in self._pazzo_steps:
            self._pazzo_steps[opp_i] = 0
            self._pazzo_target[opp_i] = None
            self._pazzo_interval[opp_i] = int(self._rng.integers(100, 151))

        self._pazzo_steps[opp_i] += 1

        if self._pazzo_target[opp_i] is None or self._pazzo_steps[opp_i] > self._pazzo_interval[opp_i]:
            self._pazzo_target[opp_i] = (
                self._rng.uniform(-self.HW, self.HW),
                self._rng.uniform(-self.HH, self.HH)
            )
            self._pazzo_steps[opp_i] = 0
            self._pazzo_interval[opp_i] = int(self._rng.integers(100, 151))

        # Nếu bóng ở trung tâm: jitter ngẫu nhiên ngắm mục tiêu
        if abs(b.x) < 5.0 and abs(b.y) < 5.0 and math.hypot(b.xs, b.ys) < 0.5:
            target_x = self._rng.uniform(-150.0, 150.0)
            target_y = self._rng.uniform(-150.0, 150.0)
        else:
            target_x, target_y = self._pazzo_target[opp_i]

        comp_x = target_x - ag.x
        comp_y = target_y - ag.y
        vers_denom = math.hypot(comp_x, comp_y)

        hor, ver = 0.0, 0.0

        if vers_denom > 40.0:
            prob_x = abs(comp_x) / vers_denom
            prob_y = abs(comp_y) / vers_denom
            if prob_x > prob_y: prob_x = 1.0
            elif prob_y > prob_x: prob_y = 1.0

            if self._rng.random() < prob_x:
                hor = 1.0 if comp_x > 0 else -1.0
            if self._rng.random() < prob_y:
                ver = 1.0 if comp_y > 0 else -1.0

        return (hor, ver, 0)

    def _get_wanderer_action(self, ag, opp_i):
        """WandererBot: every rand[3,12] steps picks a point within radius 30 of the ball
        and moves toward it. Kicks ~30% when in range."""
        b  = self.ball

        dist_to_ball = math.hypot(b.x - ag.x, b.y - ag.y)
        in_range = dist_to_ball - ag.radius - b.radius < KICK_RANGE

        # Initialise persistent state
        if not hasattr(self, '_wanderer_steps'):
            self._wanderer_steps = {}
            self._wanderer_interval = {}
            self._wanderer_target = {}
            
        if opp_i not in self._wanderer_steps:
            self._wanderer_steps[opp_i] = 0
            self._wanderer_interval[opp_i] = int(self._rng.integers(1, 4))
            self._wanderer_target[opp_i] = (b.x, b.y)

        self._wanderer_steps[opp_i] += 1

        # Pick a new target every rand[3,12] steps
        if self._wanderer_steps[opp_i] >= self._wanderer_interval[opp_i]:
            angle = float(self._rng.uniform(0, 2 * math.pi))
            r     = float(self._rng.uniform(0, 60.0))
            tx = b.x + r * math.cos(angle)
            ty = b.y + r * math.sin(angle)
            # Clamp inside field
            tx = max(-self.HW + PLYR_R, min(self.HW - PLYR_R, tx))
            ty = max(-self.HH + PLYR_R, min(self.HH - PLYR_R, ty))
            self._wanderer_target[opp_i] = (tx, ty)
            self._wanderer_steps[opp_i] = 0
            self._wanderer_interval[opp_i] = int(self._rng.integers(1, 4))

        target_x, target_y = self._wanderer_target[opp_i]
        dx = target_x - ag.x
        dy = target_y - ag.y
        dist = math.hypot(dx, dy)

        if dist < 1.0:
            hor, ver = 0.0, 0.0
        else:
            dx /= dist; dy /= dist
            best_idx, best_dot = 0, -2.0
            for i, (mx, my) in enumerate(DIR_MAP):
                dot = dx * mx + dy * my
                if dot > best_dot:
                    best_dot = dot; best_idx = i
            hor, ver = DIR_MAP[best_idx]

        # Kick occasionally when in range (~30% chance per step)
        kick = 1 if in_range and self._rng.random() < 0.10 else 0

        return (hor, ver, kick)

    def _get_defender_action(self, ag):
        """Block the direct line between ball and defender's own goal center.
        Retreats to defensive line when ball is far.
        """
        b = self.ball
        
        # Opponent's goal x coordinate (they defend this)
        my_goal_x = -self.HW if self.team_id == 1 else self.HW
        
        # Intercept point:
        target_x = max(min(b.x + b.xs * 15, self.HW), -self.HW)
        target_y = max(min(b.y + b.ys * 15, self.HH), -self.HH)
        
        # Limit target x to defensive half
        if self.team_id == 1: # Red is on left, Blue (opp) is on right. BLUE defends HW.
            target_x = max(0, min(target_x, self.HW - 50))
        else: # RED (opp) defends -HW.
            target_x = min(0, max(target_x, -self.HW + 50))
            
        # Target goal center
        dx, dy = target_x - ag.x, target_y - ag.y
        dist = math.hypot(dx, dy)
        kick = 1 if (math.hypot(b.x - ag.x, b.y - ag.y) - ag.radius - b.radius < KICK_RANGE) else 0
        
        if dist < 5.0:
            return (0, 0, kick)
            
        dx, dy = dx/dist, dy/dist
        best_dir_idx = 0
        best_dot = -2.0
        for i, (mx, my) in enumerate(DIR_MAP):
            dot = dx*mx + dy*my
            if dot > best_dot:
                best_dot = dot
                best_dir_idx = i
                
        out_dx, out_dy = DIR_MAP[best_dir_idx]
        return (out_dx, out_dy, kick)

    def render(self):
        pass

    def action_masks(self) -> np.ndarray:
        """
        Returns a 1D boolean array of shape (11,) for MultiDiscrete([9, 2]).
        Masks out the kick action (index 10) if the ball is out of range.
        """
        agent = self.agents[0]
        bx, by = self.ball.x, self.ball.y
        dist_diff = math.hypot(agent.x - bx, agent.y - by) - PLYR_R - BALL_R - KICK_RANGE
        can_kick = dist_diff <= 0
        
        mask = np.ones(11, dtype=bool)
        mask[10] = can_kick
        return mask

# ─────────────────────────────────────────────────────────────────────────────
# Standalone physics helper (module-level for speed)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_dd(da: Disc, db: Disc) -> bool:
    """Exact port of resolveDDCollision from test_index.html. Returns True if collision occurred."""
    ddx = da.x - db.x; ddy = da.y - db.y
    dist = math.hypot(ddx, ddy)
    r_sum = da.radius + db.radius
    if 0 < dist <= r_sum:
        nx, ny  = ddx / dist, ddy / dist
        
        # Handle immovable objects (Pole has imass == 0)
        imass_sum = da.imass + db.imass
        if imass_sum == 0:
            return True # both immovable, do nothing
            
        mf = da.imass / imass_sum
        overlap = r_sum - dist

        # Position correction
        da.x += nx * overlap * mf;       da.y += ny * overlap * mf
        db.x -= nx * overlap * (1 - mf); db.y -= ny * overlap * (1 - mf)

        # Momentum transfer
        rvn = (da.xs - db.xs) * nx + (da.ys - db.ys) * ny
        if rvn < 0:
            impulse = rvn * (da.bcoef * db.bcoef + 1)
            da.xs -= nx * impulse * mf;       da.ys -= ny * impulse * mf
            db.xs += nx * impulse * (1 - mf); db.ys += ny * impulse * (1 - mf)

        return True
    return False

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
import gymnasium as gym
from gymnasium import spaces

# ─────────────────────────────────────────────────────────────────────────────
# Physics constants (must match test_index.html exactly)
# ─────────────────────────────────────────────────────────────────────────────
BALL_R        = 5.8
BALL_DAMP     = 0.99
BALL_BCOEF    = 0.412
BALL_IMASS    = 1.5

PLYR_R        = 15.0
PLYR_DAMP     = 0.96
PLYR_IMASS    = 0.5
PLYR_BCOEF    = 0.5
PLYR_ACC      = 0.11
PLYR_KICK_ACC = 0.083
KICK_STR      = 4.545
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
OBS_DIM   = 4 + 4 + 4 + 11 + 2 + N_TM * 9 + N_OPP * 9 + 2 + N_TM + 3 # = 115 (added nearest tm dist)

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

    def __init__(self, phase: str = 'A0', n_agents: int = 1, seed: Optional[int] = None, legacy_obs: bool = False):
        super().__init__()

        self.phase = phase
        self.n_agents = n_agents
        self.legacy_obs = legacy_obs
        self._rng = np.random.default_rng(seed)
        
        self.frame_skip = 3 if self.phase in ('A1.2', 'A0.1', 'A1') else FRAME_SKIP

        # Gymnasium spaces
        self.obs_dim = 100 if self.legacy_obs else OBS_DIM
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
        self.last_touch = None        # 'A' for Agent, 'O' for Opponent
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
        self._prev_ball_speed   = 0.0

    # ─── reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # ── Time Curriculum ──
        if self.phase == 'A0':
            ep_secs = 15 # 15 seconds (150 steps)
        elif self.phase == 'A0.1':
            ep_secs = 30 # 30 seconds
        else:
            ep_secs = 90 # 1.5 minutes
        
        self.max_steps = ep_secs * PHYSICS_HZ // self.frame_skip
        self.step_count = 0
        self.scores = [0, 0]  # Reset scores every episode

        # Randomize team each episode → shared policy via x-flip
        self.team_id      = int(self._rng.integers(1, 3))   # 1 or 2
        self._flip        = 1.0 if self.team_id == 1 else -1.0
        self._attack_sign = 1   if self.team_id == 1 else -1

        self._reset_positions()
        
        # Init dense-reward tracking
        a = self.agents[0]
        self._prev_dist_to_ball = math.hypot(a.x - self.ball.x, a.y - self.ball.y)
        self._min_dist_to_ball = self._prev_dist_to_ball
        self._closest_agent_to_ball = self._prev_dist_to_ball
        
        goal_x = self.HW * self._attack_sign
        self._prev_ball_dist_to_goal = _dist_to_goal_segment(
            self.ball.x, self.ball.y,
            goal_x, self.goal_y, -self.goal_y
        )
        self._closest_ball_to_goal = self._prev_ball_dist_to_goal
        
        self._prev_ball_x       = self.ball.x
        self._prev_ball_speed   = math.hypot(self.ball.xs, self.ball.ys)
        self.last_touch = None
        self._a0_1_best_ball_dist_to_goal = self._prev_ball_dist_to_goal
        self._a0_1_steps_since_ball_record = 0
        self._a0_1_record_ball_to_goal_reward_total = 0.0

        return self._get_obs(), {}

    def _get_num_opp_bypassed(self):
        if not self.agents: return 0
        atk = self._attack_sign
        opps = self.agents[self.n_agents:]
        return sum(1 for opp in opps if (self.ball.x - opp.x) * atk > 0)

    def _reset_positions(self):
        size_class = '1v1'
        cands = [p for p in MAP_PRESETS if p[3] == size_class] or MAP_PRESETS
        preset = cands[int(self._rng.integers(0, len(cands)))]
        
        if self.phase == 'A0':
            # Field: 80% to 120%
            scale_h = float(self._rng.uniform(0.8, 1.2))
            scale_w = float(self._rng.uniform(0.8, 1.2))
            self.HH = float(preset[1]) * scale_h
            self.HW = float(preset[0]) * scale_w
            if self.HH > self.HW:
                self.HH, self.HW = self.HW, self.HH
                
            # Goal: 40% to 140% of preset (train with both narrow & wide goals)
            self.goal_y = float(self._rng.uniform(0.4, 1.4)) * float(preset[2])
            
            # Goal center
            padding = 10.0
            max_center = max(0.0, self.HH - self.goal_y - padding)
            self.goal_center_y = float(self._rng.uniform(-max_center, max_center))
            
            # Spawn ball (speed 0)
            bx = float(self._rng.uniform(-self.HW * 0.7, self.HW * 0.7))
            by = float(self._rng.uniform(-self.HH * 0.7, self.HH * 0.7))
            self.ball = Disc(bx, by, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
            
            self.agents = []
            pos = self._safe_spawn()
            self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
            
        else: # A0.1, A1 or A1.2
            self.HH = float(preset[1])
            self.HW = float(preset[0])
            self.goal_center_y = 0.0

            def a1_2_goal_y() -> float:
                t = getattr(self, 'total_timesteps_elapsed', 0)
                if t < 2_000_000:
                    upper_bound = self.HH
                elif t < 3_000_000:
                    phase_progress = (t - 2_000_000) / 1_000_000.0
                    upper_bound = self.HH - (self.HH - 64.0) * phase_progress
                else:
                    upper_bound = 64.0
                return float(self._rng.uniform(50.0, upper_bound))

            def a0_1_goal_y() -> float:
                t = min(getattr(self, 'total_timesteps_elapsed', 0), CURRICULUM_A0_1_STEPS)
                if t >= CURRICULUM_A0_1_STEPS:
                    return 64.0
                progress = t / CURRICULUM_A0_1_STEPS
                upper_bound = 100.0 - (100.0 - 64.0) * progress  # 100 → 64
                return float(self._rng.uniform(50.0, upper_bound))

            # ── Choose episode type ───────────────────────────────────────────
            if self.forced_opponent_type is not None:
                # External override: always use this opponent type (e.g. 'Human' from eval_render)
                self.episode_type    = 'opponent'
                self.opponent_type   = self.forced_opponent_type
                self.goal_y = 64.0
                # Only clear policy if not Trained — eval scripts may inject opponent_policy externally
                if self.forced_opponent_type != 'Trained':
                    self.opponent_policy = None
            elif self.phase == 'A0.1':
                self.episode_type = 'opponent'
                if getattr(self, 'total_timesteps_elapsed', 0) < 5_000_000:
                    r = self._rng.random()
                    if r < 0.5:
                        self.opponent_type = 'Random'
                    else:
                        self.opponent_type = 'Pazzo'
                else:
                    r = self._rng.random()
                    if r < 0.333:
                        self.opponent_type = 'Wanderer'
                    elif r < 0.666:
                        self.opponent_type = 'Pazzo'
                    else:
                        self.opponent_type = 'Random'
                self.opponent_policy = None
                self.goal_y = 64.0
                # team_id / _flip / _attack_sign already randomized in reset()
                
            elif self.phase == 'A1':
                # Self-Play phase: map always 1v1, goal always 64
                self.episode_type = 'opponent'
                self.goal_y = 64.0
                self.opponent_type = 'Trained'
                # Only reset policy if not externally forced (e.g. eval/play scripts set forced_opponent_type)
                if self.forced_opponent_type is None:
                    self.opponent_policy = None
                
            elif self._rng.random() < A1_PRECISION_RATIO:
                # PRECISION episode: small-medium goal, no opponent (20%)
                self.episode_type  = 'precision'
                scale = float(self._rng.uniform(0.4, 1.0))
                self.goal_y        = float(preset[2]) * scale
                self.opponent_type = 'None'
                self.opponent_policy = None
            else:
                # OPPONENT episode: large→small goal curriculum
                self.episode_type = 'opponent'
                t        = min(self.total_timesteps_elapsed, A1_OPP_CURRICULUM_STEPS)
                progress = t / A1_OPP_CURRICULUM_STEPS
                scale    = A1_OPP_GOAL_START + (A1_OPP_GOAL_END - A1_OPP_GOAL_START) * progress
                self.goal_y = float(preset[2]) * scale

                # Opponent distribution (within 80% opponent episodes):
                # 25% Defender | 15% Attacker | 30% A0 frozen | 30% Hybrid
                r = self._rng.random()
                if r < 0.25:
                    self.opponent_type   = 'Defender'
                    self.opponent_policy = None
                elif r < 0.40:
                    self.opponent_type   = 'Attacker'
                    self.opponent_policy = None
                elif r < 0.70:
                    if self.a0_model_path is not None:
                        from stable_baselines3 import PPO as _PPO
                        self.opponent_type   = 'Trained'
                        self.opponent_policy = _PPO.load(self.a0_model_path, device='cpu')
                    else:
                        self.opponent_type   = 'Defender'  # fallback if no A0 model
                        self.opponent_policy = None
                else:
                    self.opponent_type   = 'Hybrid'
                    self.opponent_policy = None
                    self._hybrid_mode    = 'follower'  # reset hybrid state each episode

            # ── Positions ─────────────────────────────────────────────────────
            self.agents = []
            if self.episode_type == 'precision' and self.phase != 'A0.1':
                # No opponent — just agent
                self.ball = Disc(0.0, 0.0, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
                pos = self._safe_spawn()
                self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
            else:
                if self.phase == 'A0.1':
                    # Spawn ball randomly anywhere on the field
                    bx = float(self._rng.uniform(-self.HW + BALL_R, self.HW - BALL_R))
                    by = float(self._rng.uniform(-self.HH + BALL_R, self.HH - BALL_R))
                    self.ball = Disc(bx, by, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)

                    # agent's own half = OPPOSITE to attack direction
                    # opponent's half  = SAME side as attack goal
                    atk = self._attack_sign  # +1 RED attacks right, -1 BLUE attacks left
                    if atk == 1:  # RED attacks right (+HW): agent left half, opp right half
                        agent_x_min, agent_x_max = -self.HW * 0.85 + PLYR_R, -PLYR_R
                        opp_x_min,   opp_x_max   = PLYR_R, self.HW * 0.85 - PLYR_R
                    else:         # BLUE attacks left (-HW): agent right half, opp left half
                        agent_x_min, agent_x_max = PLYR_R, self.HW * 0.85 - PLYR_R
                        opp_x_min,   opp_x_max   = -self.HW * 0.85 + PLYR_R, -PLYR_R

                    # Spawn agent in own half
                    pos = self._safe_spawn(x_min=agent_x_min, x_max=agent_x_max)
                    self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))

                    # Spawn opponent in their own half (= agent's attack half)
                    pos2 = self._safe_spawn(x_min=opp_x_min, x_max=opp_x_max)
                    self.agents.append(Disc(pos2[0], pos2[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))

                elif self.phase == 'A1.2':
                    # Randomize players in their respective halves
                    red_x = float(self._rng.uniform(-self.HW + PLYR_R, 0 - PLYR_R))
                    red_y = float(self._rng.uniform(-self.HH + PLYR_R, self.HH - PLYR_R))
                    blue_x = float(self._rng.uniform(0 + PLYR_R, self.HW - PLYR_R))
                    blue_y = float(self._rng.uniform(-self.HH + PLYR_R, self.HH - PLYR_R))

                    if self.team_id == 1:
                        self.agents.append(Disc(red_x, red_y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Agent (RED)
                        self.agents.append(Disc(blue_x, blue_y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Opp (BLUE)
                    else:
                        self.agents.append(Disc(blue_x, blue_y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Agent (BLUE)
                        self.agents.append(Disc(red_x, red_y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Opp (RED)

                    # Spawn ball uniformly anywhere on the field
                    bx = float(self._rng.uniform(-self.HW + BALL_R, self.HW - BALL_R))
                    by = float(self._rng.uniform(-self.HH + BALL_R, self.HH - BALL_R))
                    self.ball = Disc(bx, by, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
                    
                elif self.phase == 'A2.0':
                    self.episode_type = 'opponent'
                    self.goal_y = 64.0
                    self.opponent_type = self.forced_opponent_type if isinstance(self.forced_opponent_type, list) else ('Trained' if self.forced_opponent_type is None else self.forced_opponent_type)
                    
                    our_sign = self._attack_sign
                    our_x_min = -self.HW * 0.85 + PLYR_R if our_sign == 1 else PLYR_R
                    our_x_max = -PLYR_R if our_sign == 1 else self.HW * 0.85 - PLYR_R
                    
                    opp_x_min = PLYR_R if our_sign == 1 else -self.HW * 0.85 + PLYR_R
                    opp_x_max = self.HW * 0.85 - PLYR_R if our_sign == 1 else -PLYR_R

                    self.ball = Disc(float(self._rng.uniform(-self.HW + BALL_R, self.HW - BALL_R)),
                                     float(self._rng.uniform(-self.HH + BALL_R, self.HH - BALL_R)),
                                     0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
                    self.agents = []
                    
                    if self.n_agents == 2:
                        pos = self._safe_spawn(x_min=our_x_min, x_max=our_x_max)
                        self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
                        pos = self._safe_spawn(x_min=our_x_min, x_max=our_x_max)
                        self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
                        pos = self._safe_spawn(x_min=opp_x_min, x_max=opp_x_max)
                        self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
                    else:
                        pos = self._safe_spawn(x_min=our_x_min, x_max=our_x_max)
                        self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
                        pos = self._safe_spawn(x_min=opp_x_min, x_max=opp_x_max)
                        self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
                        pos = self._safe_spawn(x_min=opp_x_min, x_max=opp_x_max)
                        self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))

                elif self.phase == 'A2v2':
                    self.episode_type = 'opponent'
                    self.goal_y = 64.0
                    self.opponent_type = self.forced_opponent_type if isinstance(self.forced_opponent_type, list) else ('Trained' if self.forced_opponent_type is None else self.forced_opponent_type)
                    
                    our_sign = self._attack_sign
                    our_x_min = -self.HW * 0.85 + PLYR_R if our_sign == 1 else PLYR_R
                    our_x_max = -PLYR_R if our_sign == 1 else self.HW * 0.85 - PLYR_R
                    
                    opp_x_min = PLYR_R if our_sign == 1 else -self.HW * 0.85 + PLYR_R
                    opp_x_max = self.HW * 0.85 - PLYR_R if our_sign == 1 else -PLYR_R

                    self.ball = Disc(float(self._rng.uniform(-self.HW + BALL_R, self.HW - BALL_R)),
                                     float(self._rng.uniform(-self.HH + BALL_R, self.HH - BALL_R)),
                                     0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
                    self.agents = []
                    
                    pos = self._safe_spawn(x_min=our_x_min, x_max=our_x_max)
                    self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
                    pos = self._safe_spawn(x_min=our_x_min, x_max=our_x_max)
                    self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
                    pos = self._safe_spawn(x_min=opp_x_min, x_max=opp_x_max)
                    self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
                    pos = self._safe_spawn(x_min=opp_x_min, x_max=opp_x_max)
                    self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
                        
                else:
                    self.ball = Disc(0.0, 0.0, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
                    
                    # y random trong -goal_y, goal_y
                    y1 = float(self._rng.uniform(-self.goal_y, self.goal_y))
                    y2 = float(self._rng.uniform(-self.goal_y, self.goal_y))
                    
                    # Khoảng cách tới bóng của người gần hơn
                    dist_near = float(self._rng.uniform(PLYR_R + BALL_R + 5.0, self.HW * 0.5))
                    # Khoảng cách tới bóng của người xa hơn (hơn người kia từ 60 đến 90)
                    dist_far = dist_near + float(self._rng.uniform(60.0, 90.0))
                    
                    # Random xem RED hay BLUE ở gần bóng hơn
                    if self._rng.random() < 0.5:
                        d_red = dist_near
                        d_blue = dist_far
                    else:
                        d_red = dist_far
                        d_blue = dist_near
                        
                    # Red ở bên trái (âm), Blue ở bên phải (dương)
                    red_x = -d_red
                    blue_x = d_blue
                    
                    if self.team_id == 1:
                        self.agents.append(Disc(red_x, y1, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Agent (RED)
                        self.agents.append(Disc(blue_x, y2, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Opp (BLUE)
                    else:
                        self.agents.append(Disc(blue_x, y2, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Agent (BLUE)
                        self.agents.append(Disc(red_x, y1, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Opp (RED)

        if self.phase.startswith('A2'):
            self.a2_adv = 0

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
    def _get_bot_action(self, bot_type, ag_idx):
        if bot_type == 'Defender':
            return self._get_defender_action(ag_idx)
        elif bot_type == 'Attacker':
            return self._get_attacker_action(ag_idx)
        elif bot_type == 'Hybrid':
            if self.last_touch == 'A':
                self._hybrid_mode = 'defender'
            elif self.last_touch == 'O':
                self._hybrid_mode = 'follower'
            if getattr(self, '_hybrid_mode', 'defender') == 'defender':
                return self._get_defender_action(ag_idx)
            else:
                return self._get_follower_action(ag_idx)
        elif bot_type == 'Follower':
            return self._get_follower_action(ag_idx)
        elif bot_type == 'Pazzo':
            return self._get_pazzo_action(ag_idx)
        elif bot_type == 'Wanderer':
            return self._get_wanderer_action(ag_idx)
        elif bot_type == 'Static':
            return self._get_static_action(ag_idx)
        else:
            return self._get_random_action(ag_idx)

    def step(self, action: np.ndarray):
        assert self.ball is not None, "Call reset() before step()"

        dir_idx = int(action[0])
        kick    = int(action[1])
        dx, dy  = DIR_MAP[dir_idx]
        
        agent_actions = [(dx * self._flip, dy, kick)]
        
        if self.phase in ('A1', 'A1.2', 'A0.1', 'A2.0', 'A2v2') and len(self.agents) > 1:
            if self.phase == 'A2.0':
                if self.n_agents == 2:
                    # Teammate (agents[1])
                    if hasattr(self, 'teammate_policy') and self.teammate_policy is not None:
                        tm_obs = self._get_obs_for_teammate()
                        expected_shape = self.teammate_policy.observation_space.shape[0]
                        tm_action, _ = self.teammate_policy.predict(tm_obs[:expected_shape], deterministic=False)
                        tm_dx, tm_dy = DIR_MAP[int(tm_action[0])]
                        agent_actions.append((tm_dx * self._flip, tm_dy, int(tm_action[1])))
                    else:
                        agent_actions.append((0.0, 0.0, 0))
                        
                    # Opponent (agents[2])
                    if self.opponent_type == 'Human':
                        agent_actions.append(self.human_opponent_action)
                    elif self.opponent_policy is not None:
                        opp_obs = self._get_obs_for_opponent(opp_idx=2)
                        expected_shape = self.opponent_policy.observation_space.shape[0]
                        opp_action, _ = self.opponent_policy.predict(opp_obs[:expected_shape], deterministic=False)
                        opp_dx, opp_dy = DIR_MAP[int(opp_action[0])]
                        agent_actions.append((opp_dx * (-self._flip), opp_dy, int(opp_action[1])))
                    else:
                        agent_actions.append(self._get_bot_action(self.opponent_type, 2))
                else:
                    # Opponent 1 (agents[1])
                    opp1_type = self.opponent_type[0] if isinstance(self.opponent_type, list) else self.opponent_type
                    if opp1_type == 'Human':
                        agent_actions.append(self.human_opponent_action)
                    elif self.opponent_policy is not None:
                        opp_obs = self._get_obs_for_opponent(opp_idx=1)
                        expected_shape = self.opponent_policy.observation_space.shape[0]
                        opp_action, _ = self.opponent_policy.predict(opp_obs[:expected_shape], deterministic=False)
                        opp_dx, opp_dy = DIR_MAP[int(opp_action[0])]
                        agent_actions.append((opp_dx * (-self._flip), opp_dy, int(opp_action[1])))
                    else:
                        agent_actions.append(self._get_bot_action(opp1_type, 1))
                        
                    # Opponent 2 (agents[2])
                    opp2_type = self.opponent_type[1] if isinstance(self.opponent_type, list) else self.opponent_type
                    if self.opponent_policy is not None:
                        opp_obs = self._get_obs_for_opponent(opp_idx=2)
                        expected_shape = self.opponent_policy.observation_space.shape[0]
                        opp_action, _ = self.opponent_policy.predict(opp_obs[:expected_shape], deterministic=False)
                        opp_dx, opp_dy = DIR_MAP[int(opp_action[0])]
                        agent_actions.append((opp_dx * (-self._flip), opp_dy, int(opp_action[1])))
                    else:
                        agent_actions.append(self._get_bot_action(opp2_type, 2))
            elif self.phase == 'A2v2':
                # Teammate (agents[1])
                if hasattr(self, 'teammate_policy') and self.teammate_policy is not None:
                    tm_obs = self._get_obs_for_teammate()
                    expected_shape = self.teammate_policy.observation_space.shape[0]
                    tm_action, _ = self.teammate_policy.predict(tm_obs[:expected_shape], deterministic=False)
                    tm_dx, tm_dy = DIR_MAP[int(tm_action[0])]
                    agent_actions.append((tm_dx * self._flip, tm_dy, int(tm_action[1])))
                else:
                    agent_actions.append((0.0, 0.0, 0))
                    
                # Opponent 1 (agents[2])
                opp1_type = self.opponent_type[0] if isinstance(self.opponent_type, list) else self.opponent_type
                if opp1_type == 'Human':
                    agent_actions.append(self.human_opponent_action)
                elif self.opponent_policy is not None:
                    opp_obs = self._get_obs_for_opponent(opp_idx=2)
                    expected_shape = self.opponent_policy.observation_space.shape[0]
                    opp_action, _ = self.opponent_policy.predict(opp_obs[:expected_shape], deterministic=False)
                    opp_dx, opp_dy = DIR_MAP[int(opp_action[0])]
                    agent_actions.append((opp_dx * (-self._flip), opp_dy, int(opp_action[1])))
                else:
                    agent_actions.append(self._get_bot_action(opp1_type, 2))
                    
                # Opponent 2 (agents[3])
                opp2_type = self.opponent_type[1] if isinstance(self.opponent_type, list) else self.opponent_type
                if self.opponent_policy is not None:
                    opp_obs = self._get_obs_for_opponent(opp_idx=3)
                    expected_shape = self.opponent_policy.observation_space.shape[0]
                    opp_action, _ = self.opponent_policy.predict(opp_obs[:expected_shape], deterministic=False)
                    opp_dx, opp_dy = DIR_MAP[int(opp_action[0])]
                    agent_actions.append((opp_dx * (-self._flip), opp_dy, int(opp_action[1])))
                else:
                    agent_actions.append(self._get_bot_action(opp2_type, 3))
            else:
                if self.opponent_type == 'Defender':
                    agent_actions.append(self._get_defender_action())
                elif self.opponent_type == 'Attacker':
                    agent_actions.append(self._get_attacker_action())
                elif self.opponent_type == 'Hybrid':
                    # Switch mode based on who last touched the ball
                    if self.last_touch == 'A':   # agent has ball → defend
                        self._hybrid_mode = 'defender'
                    elif self.last_touch == 'O': # opponent touched ball → press
                        self._hybrid_mode = 'follower'
                    if self._hybrid_mode == 'defender':
                        agent_actions.append(self._get_defender_action())
                    else:
                        agent_actions.append(self._get_follower_action())
                elif self.opponent_type == 'Trained' and self.opponent_policy is not None:
                    # Build obs from opponent's perspective
                    opp_obs = self._get_obs_for_opponent()
                    expected_shape = self.opponent_policy.observation_space.shape[0]
                    opp_action, _ = self.opponent_policy.predict(opp_obs[:expected_shape], deterministic=False)
                    opp_dir_idx = int(opp_action[0])
                    opp_kick = int(opp_action[1])
                    opp_dx, opp_dy = DIR_MAP[opp_dir_idx]
                    
                    # The network output is relative to the opponent's perspective.
                    # Since the learning agent uses self._flip, the opponent uses -self._flip
                    opp_flip = -self._flip
                    agent_actions.append((opp_dx * opp_flip, opp_dy, opp_kick))
                elif self.opponent_type == 'Human':
                    agent_actions.append(self.human_opponent_action)
                elif self.opponent_type == 'Static':
                    agent_actions.append(self._get_static_action())
                elif self.opponent_type == 'Random':
                    agent_actions.append(self._get_random_action())
                elif self.opponent_type == 'Pazzo':
                    agent_actions.append(self._get_pazzo_action())
                elif self.opponent_type == 'Wanderer':
                    agent_actions.append(self._get_wanderer_action())
                else:
                    agent_actions.append((0.0, 0.0, 0)) # Default/None

        # Run self.frame_skip ticks
        goal_result = 0   # 0=none, 1=left(own for red), 2=right(scored for red)
        
        # Track touches for the whole step
        touch_events = []
        
        for _ in range(self.frame_skip):
            self.last_touch_this_tick = None
            self.last_touch_idx_this_tick = None
            result = self._tick(agent_actions)
            if self.last_touch_this_tick is not None:
                touch_events.append(self.last_touch_this_tick)
                if hasattr(self, 'last_touch_idx_this_tick') and self.last_touch_idx_this_tick is not None:
                    self.last_touch_idx = self.last_touch_idx_this_tick
                
            if result != 0:
                goal_result = result
                break
        # Preserve previous last_touch for farming checks, then update
        prev_last_touch = self.last_touch
        if touch_events:
            self.last_touch = touch_events[-1]

        self.step_count += 1

        # Snapshot after move
        a = self.agents[0]
        cur_dist     = math.hypot(a.x - self.ball.x, a.y - self.ball.y)
        cur_ball_spd = math.hypot(self.ball.xs, self.ball.ys)
        atk = self._attack_sign
        cur_ball_dist_to_goal = _dist_to_goal_segment(
            self.ball.x, self.ball.y,
            self.HW * atk, self.goal_y, -self.goal_y
        )
        # (approach_ball_record_delta and delta_ball_to_goal removed — A0.1 no longer uses them)

        # ── Reward ────────────────────────────────────────────────────────────
        reward     = 0.0
        terminated = False
        truncated = False

        if self.phase == 'A0':
            if goal_result == 2:
                # Scored into right goal ✅
                reward = 5.0
                terminated = True
            else:
                reward -= 0.003 # Dense punishment per step
                
            if not terminated and self.step_count >= self.max_steps:
                truncated = True

        elif not self.phase.startswith('A2'): # A1, A1.2, or A0.1 Phase
            # Goal logic
            goal_scored = False
            if goal_result == 2:
                # Agent scored
                self.scores[self.team_id - 1] += 1
                reward += 30.0
                goal_scored = True
                self._reset_positions()
            elif goal_result == 1:
                # Opponent scored
                opp_id = 2 if self.team_id == 1 else 1
                self.scores[opp_id - 1] += 1
                reward -= 10.0
                goal_scored = True
                self._reset_positions()

            # Penalty/Reward logic for A1.2
            if self.phase == 'A1.2' and not goal_scored:
                # Dense reward for ball advancing toward goal
                cur_ball_dist_to_goal = _dist_to_goal_segment(
                    self.ball.x, self.ball.y,
                    self.HW * atk, self.goal_y, -self.goal_y
                )
                delta_dist = self._prev_ball_dist_to_goal - cur_ball_dist_to_goal
                reward += delta_dist * (1.0 / (2.0 * self.HW))

                field_diag = math.hypot(self.HW * 2.0, self.HH * 2.0)
                # Penalty khi đứng xa bóng với slope = 2
                dist_penalty = (cur_dist / field_diag) * 2.0
                reward -= dist_penalty
                
            if self.phase == 'A0.1':
                # ── Reset reward and re-apply terminal signals cleanly ─────────
                reward = 0.0
                if goal_result == 2:
                    # Ball entered agent's attack goal.
                    # Only reward if agent was the last to touch the ball.
                    # If opponent scored into their own goal (last_touch == 'O'), no bonus.
                    if self.last_touch != 'O':
                        reward += 5.0
                elif goal_result == 1:
                    # Opponent scored — re-apply penalty (was cleared by reset above)
                    reward -= 5.0

        elif self.phase.startswith('A2'):
            N = self._get_num_opp_bypassed()
            A = self.a2_adv
            touched_by_us = ('A' in touch_events)
            
            if N > A:
                if touched_by_us:
                    reward += (N - A) * 2.0
                    self.a2_adv = N
            elif N < A:
                reward += (N - A) * 2.0
                self.a2_adv = N
                
            # ── Teammate Distance Reward ───────────────────────────────────
            if self.n_agents == 2:
                dist_teammates = math.hypot(self.agents[0].x - self.agents[1].x, self.agents[0].y - self.agents[1].y)
                if dist_teammates > 80.0:
                    reward += 0.002
                    
            # ── Ball Speed Penalty ─────────────────────────────────────────
            ball_speed = math.hypot(self.ball.xs, self.ball.ys)
            if ball_speed < 0.2:
                reward -= 0.003
                
            # ── Goal Logic ────────────────────────────────────────────────
            if goal_result == 2: # Agent scored
                self.scores[self.team_id - 1] += 1
                time_bonus = 0.002 * max(0, self.max_steps - self.step_count)
                reward += 10.0 + time_bonus
                goal_scored = True
                self._reset_positions()
                
            elif goal_result == 1: # Opponent scored
                opp_id = 2 if self.team_id == 1 else 1
                self.scores[opp_id - 1] += 1
                
                reward -= 10.0
                goal_scored = True
                self._reset_positions()

        # Update previous tracking
        self._prev_dist_to_ball = math.hypot(a.x - self.ball.x, a.y - self.ball.y)
        goal_x = self.HW * atk
        self._prev_ball_dist_to_goal = _dist_to_goal_segment(
            self.ball.x, self.ball.y,
            goal_x, self.goal_y, -self.goal_y
        )
        self._prev_ball_speed = math.hypot(self.ball.xs, self.ball.ys)

        # ── Global Terminal Logic ──────────────────────────────────────────────
        if self.phase != 'A0':
            if self.scores[0] >= 1 or self.scores[1] >= 1:
                terminated = True
                
        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        info = {
            "a0_1/ball_to_goal_record_reward_total": float(self._a0_1_record_ball_to_goal_reward_total),
            "a0_1/best_ball_dist_to_goal": float(self._a0_1_best_ball_dist_to_goal),
            "a0_1/cur_ball_dist_to_goal": float(cur_ball_dist_to_goal),
            "a0_1/steps_since_ball_record": float(self._a0_1_steps_since_ball_record),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ─── Physics ──────────────────────────────────────────────────────────────
    def _tick(self, agent_actions) -> int:
        """
        One physics tick. Takes a list of (dx, dy, kick) for each agent.
        Returns 0=normal | 1=left-goal | 2=right-goal.
        """
        ball   = self.ball
        agents = self.agents
        HW, HH = self.HW, self.HH

        # Randomize agent action order each tick to prevent systematic first-mover advantage
        action_order = list(range(len(agent_actions)))
        self._rng.shuffle(action_order)

        # 1. Kick (before movement) — random order
        kicked_agents = []
        for i in action_order:
            ag = agents[i]
            dx, dy, kick = agent_actions[i]
            if kick:
                dx_b = ball.x - ag.x
                dy_b = ball.y - ag.y
                dist = math.hypot(dx_b, dy_b)
                if dist > 0 and dist - ag.radius - ball.radius < KICK_RANGE:
                    nx, ny = dx_b / dist, dy_b / dist
                    ball.xs += nx * KICK_STR
                    ball.ys += ny * KICK_STR
                    kicked_agents.append(i)

        # 2. Acceleration — random order
        for i in action_order:
            ag = agents[i]
            dx, dy, kick = agent_actions[i]
            ln = math.hypot(dx, dy)
            acc = PLYR_KICK_ACC if kick else PLYR_ACC
            if ln > 0:
                ndx, ndy = dx / ln, dy / ln
                ag.xs += ndx * acc
                ag.ys += ndy * acc

        # 3. Move all
        ball.x += ball.xs; ball.y += ball.ys
        for ag in agents:
            ag.x += ag.xs; ag.y += ag.ys

        # 4. Disc-Disc collisions (Players and Ball)
        # Randomize collision order each tick to prevent systematic agents[0] advantage
        shuffled_agents = list(agents)
        self._rng.shuffle(shuffled_agents)
        all_discs = [ball] + shuffled_agents
        n = len(all_discs)
        
        # Track touches in this tick
        touched_agent = False
        touched_opp = False
        last_touched_orig_idx = None
        
        # Include agents who successfully kicked the ball
        for orig_idx in kicked_agents:
            last_touched_orig_idx = orig_idx
            is_our_agent = (orig_idx < self.n_agents)
            if is_our_agent:
                touched_agent = True
            else:
                touched_opp = True

        for i in range(n):
            for j in range(i + 1, n):
                collided = _resolve_dd(all_discs[i], all_discs[j])
                if collided:
                    # Check if player hit the ball (ball is at index 0 in all_discs)
                    if (i == 0 and j > 0) or (j == 0 and i > 0):
                        disc_idx = max(i, j) - 1 # index in shuffled_agents
                        actual_agent = shuffled_agents[disc_idx]
                        orig_idx = self.agents.index(actual_agent)
                        last_touched_orig_idx = orig_idx
                        
                        is_our_agent = (orig_idx < self.n_agents)
                            
                        if is_our_agent:
                            touched_agent = True
                        else:
                            touched_opp = True

        if touched_agent and touched_opp:
            # Both touched in same tick, let's say agent wins ties.
            self.last_touch_this_tick = 'A'
        elif touched_agent:
            self.last_touch_this_tick = 'A'
        elif touched_opp:
            self.last_touch_this_tick = 'O'
        else:
            self.last_touch_this_tick = None
            
        self.last_touch_idx_this_tick = last_touched_orig_idx

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
    def _get_obs(self) -> np.ndarray:
        """Build 100-dim observation vector (matches agent-api.js getObs)."""
        ball  = self.ball
        agent = self.agents[0]
        obs   = np.zeros(OBS_DIM, dtype=np.float32)

        bx, by   = ball.x,   ball.y
        bxs, bys = ball.xs,  ball.ys
        mx, my   = agent.x,  agent.y
        mxs, mys = agent.xs, agent.ys
        flip = self._flip

        surf_dist = max(0.0, math.hypot(mx - bx, my - by) - PLYR_R - BALL_R)
        can_kick  = 1.0 if surf_dist < KICK_RANGE else 0.0

        i = 0

        # Section 1 — Field constants (4)
        obs[i] = self.goal_y / NORM;                    i += 1
        obs[i] = self.HH / NORM;                        i += 1
        obs[i] = self.HW / NORM;                        i += 1
        obs[i] = 0.0;    i += 1  # 0=RED, 1=BLUE (Always 0.0 to prevent Out-Of-Distribution for self-play opponents)

        # Section 2 — Agent ↔ Ball (4)
        obs[i] = flip * (bx - mx) / NORM;     i += 1
        obs[i] = (by - my) / NORM;            i += 1
        obs[i] = surf_dist / DIAG;            i += 1
        obs[i] = can_kick;                    i += 1

        # Section 2b — Ball ↔ Goals (4)
        top_post_y = self.goal_center_y - self.goal_y
        bot_post_y = self.goal_center_y + self.goal_y
        opp_post_y = top_post_y if abs(top_post_y - by) < abs(bot_post_y - by) else bot_post_y
        own_post_y = opp_post_y

        obs[i] = (self.HW - flip * bx) / NORM;       i += 1  # dx to opp goal line
        obs[i] = (opp_post_y - by) / NORM;           i += 1  # dy to nearest opp post
        obs[i] = (-self.HW - flip * bx) / NORM;      i += 1  # dx to own goal line
        obs[i] = (own_post_y - by) / NORM;           i += 1  # dy to nearest own post

        # Section 3 — Dynamic state (11)
        obs[i] = flip * bx / NORM;            i += 1
        obs[i] = (by - self.goal_center_y) / NORM;  i += 1 # Provide ball's relative Y to the goal center so it can aim properly when goal moves
        obs[i] = flip * bxs / MAX_SPEED;      i += 1
        obs[i] = bys / MAX_SPEED;             i += 1
        obs[i] = flip * mx / NORM;            i += 1
        obs[i] = (my - self.goal_center_y) / NORM;  i += 1
        obs[i] = flip * mxs / MAX_SPEED;      i += 1
        obs[i] = mys / MAX_SPEED;             i += 1
        obs[i] = math.hypot(mxs, mys) / MAX_SPEED; i += 1
        obs[i] = flip * (mxs - bxs) / MAX_SPEED;   i += 1 # Relative velocity X
        obs[i] = (mys - bys) / MAX_SPEED;          i += 1 # Relative velocity Y

        # Section 4 — Game state (2)
        obs[i] = max(0.0, 1.0 - self.step_count / self.max_steps); i += 1
        obs[i] = self.max_steps / MAX_STEPS_ALL_MODES;  i += 1  # episode scale (A0=0.167, A1=1.0)

        # Sections 5 & 6 — Teammates × 4 + Opponents × 5
        assert i == 25, f"Obs pointer mismatch before padding: {i}"
        
        # In 1v1 (A0.1 / A1), there are no teammates. Teammate section remains 0.
        # Section 6 (Opponents) starts at index 25 + 4*9 = 61.
        # ── Teammates ──
        tm_start_idx = 25
        teammates = self.agents[1:self.n_agents]
        for i, tm in enumerate(teammates):
            if i >= N_TM: break
            idx = tm_start_idx + i * 9
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

        # ── Opponents ──
        opp_start_idx = 25 + N_TM * 9 # 61
        opponents = self.agents[self.n_agents:]
        for i, opp in enumerate(opponents):
            if i >= N_OPP: break
            idx = opp_start_idx + i * 9
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

        # Append Advantage features at the very end (index 106 and 107)
        if self.phase.startswith('A2'):
            obs[106] = self.a2_adv / 5.0
            obs[107] = self._get_num_opp_bypassed() / 5.0

        # ── NEW GENERIC FEATURES (108 to 113) ──
        my_dist_to_ball = math.hypot(mx - bx, my - by)
        tm_idx = 108
        tm_dists_to_ball = []
        
        # 1. Distances to all teammates
        for i in range(N_TM):
            if i < len(teammates):
                tm = teammates[i]
                obs[tm_idx] = math.hypot(mx - tm.x, my - tm.y) / DIAG
                tm_dists_to_ball.append(math.hypot(tm.x - bx, tm.y - by))
            else:
                obs[tm_idx] = 0.0
            tm_idx += 1
            
        # 2. Rank, Diff, and Nearest Teammate Dist
        if not teammates:
            obs[112] = 0.0
            obs[113] = 0.0
            obs[114] = 0.0
        else:
            rank = sum(1 for d in tm_dists_to_ball if d < my_dist_to_ball)
            obs[112] = float(rank) / N_TM
            
            min_team_dist_to_ball = min(my_dist_to_ball, min(tm_dists_to_ball))
            obs[113] = (my_dist_to_ball - min_team_dist_to_ball) / DIAG
            
            my_dists_to_tms = [math.hypot(mx - tm.x, my - tm.y) for tm in teammates]
            obs[114] = min(my_dists_to_tms) / DIAG

        return obs

    def _get_obs_for_teammate(self) -> np.ndarray:
        # Swap agents[0] and agents[1] (teammate)
        self.agents[0], self.agents[1] = self.agents[1], self.agents[0]
        obs_tm = self._get_obs()
        self.agents[0], self.agents[1] = self.agents[1], self.agents[0]
        return obs_tm

    def _get_obs_for_opponent(self, opp_idx=1) -> np.ndarray:
        # Swap team context and return obs to opponent (for self play evaluation inside step)
        original_team = self.team_id
        # Swap explicitly
        old_flip = self._flip
        old_attack = self._attack_sign
        
        self.team_id = 2 if self.team_id == 1 else 1
        self._flip = 1.0 if self.team_id == 1 else -1.0
        self._attack_sign = 1 if self.team_id == 1 else -1
        
        original_agents = list(self.agents)
        
        if self.phase == 'A2v2':
            # 2v2: Swap the whole teams first
            self.agents[0], self.agents[1], self.agents[2], self.agents[3] = \
                self.agents[2], self.agents[3], self.agents[0], self.agents[1]
            if opp_idx == 3:
                self.agents[0], self.agents[1] = self.agents[1], self.agents[0]
        else:
            self.agents[0], self.agents[opp_idx] = self.agents[opp_idx], self.agents[0]
            
        obs_opp = self._get_obs()
        
        # Revert
        self.agents = original_agents
        self.team_id = original_team
        self._flip = old_flip
        self._attack_sign = old_attack
        return obs_opp

    def _get_follower_action(self, ag_idx=1):
        """Chase ball and kick toward opponent goal.
        Anti-own-goal: simulate ball trajectory after kick (ray-goal intersection).
        - If kick would enter own goal → suppress + reposition laterally.
        - Otherwise → kick freely regardless of distance.
        """
        ag = self.agents[ag_idx]
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

    def _get_attacker_action(self, ag_idx=1):
        """Aggressively tries to score into agent's goal.
        Positions behind the ball (own-goal side) so kicks go toward agent's goal.
        Has anti-own-goal ray test: never kicks into own goal.
        """
        ag = self.agents[ag_idx]
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


    def _get_static_action(self, ag_idx=1):
        """StaticModel: Always stays still"""
        return (0.0, 0.0, 0)

    def _get_random_action(self, ag_idx=1):
        """RandomModel: Randomly moves and kicks"""
        dir_idx = int(self._rng.integers(0, 9))
        kick = int(self._rng.integers(0, 2))
        dx, dy = DIR_MAP[dir_idx]
        return (dx, dy, kick)

    def _get_pazzo_action(self, ag_idx=1):
        """PazzoModel: Moves to a random point, changes every rand[100,150] steps. Jitters if ball is near center."""
        ag = self.agents[ag_idx]
        b = self.ball

        if not hasattr(self, '_pazzo_steps'):
            self._pazzo_steps = {}
            self._pazzo_target = {}
            self._pazzo_interval = {}

        if ag_idx not in self._pazzo_steps:
            self._pazzo_steps[ag_idx] = 0
            self._pazzo_target[ag_idx] = None
            self._pazzo_interval[ag_idx] = int(self._rng.integers(100, 151))

        self._pazzo_steps[ag_idx] += 1

        if self._pazzo_target[ag_idx] is None or self._pazzo_steps[ag_idx] > self._pazzo_interval[ag_idx]:
            self._pazzo_target[ag_idx] = (
                self._rng.uniform(-self.HW, self.HW),
                self._rng.uniform(-self.HH, self.HH)
            )
            self._pazzo_steps[ag_idx] = 0
            self._pazzo_interval[ag_idx] = int(self._rng.integers(100, 151))

        # Nếu bóng ở trung tâm: jitter ngẫu nhiên ngắm mục tiêu
        if abs(b.x) < 5.0 and abs(b.y) < 5.0 and math.hypot(b.xs, b.ys) < 0.5:
            target_x = self._rng.uniform(-150.0, 150.0)
            target_y = self._rng.uniform(-150.0, 150.0)
        else:
            target_x, target_y = self._pazzo_target[ag_idx]

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

    def _get_wanderer_action(self, ag_idx=1):
        """WandererBot: every rand[3,12] steps picks a point within radius 30 of the ball
        and moves toward it. Kicks ~30% when in range."""
        ag = self.agents[ag_idx]
        b  = self.ball

        dist_to_ball = math.hypot(b.x - ag.x, b.y - ag.y)
        in_range = dist_to_ball - ag.radius - b.radius < KICK_RANGE

        # Initialise persistent state
        if not hasattr(self, '_wanderer_steps'):
            self._wanderer_steps = 0
            self._wanderer_interval = int(self._rng.integers(1, 4))
            self._wanderer_target = (b.x, b.y)

        self._wanderer_steps += 1

        # Pick a new target every rand[3,12] steps
        if self._wanderer_steps >= self._wanderer_interval:
            angle = float(self._rng.uniform(0, 2 * math.pi))
            r     = float(self._rng.uniform(0, 60.0))
            tx = b.x + r * math.cos(angle)
            ty = b.y + r * math.sin(angle)
            # Clamp inside field
            tx = max(-self.HW + PLYR_R, min(self.HW - PLYR_R, tx))
            ty = max(-self.HH + PLYR_R, min(self.HH - PLYR_R, ty))
            self._wanderer_target = (tx, ty)
            self._wanderer_steps = 0
            self._wanderer_interval = int(self._rng.integers(1, 4))

        target_x, target_y = self._wanderer_target
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

    def _get_defender_action(self, ag_idx=1):
        # Stay near goal, intercept if ball comes near
        ag = self.agents[ag_idx]
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

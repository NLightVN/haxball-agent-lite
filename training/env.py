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

Reward shaping (A0):
    Dense:
        +Δ_approach_ball × 0.003     (closing in on ball)
        +Δ_ball_forward  × 0.002     (ball moving toward right goal)
        +kick_speed_delta × 0.05     (if kick caused ball speed increase)
    Terminal:
        +[5 + 3×time_bonus]          (goal scored, bonus for earlier)
        −3.0                         (own goal)
        −0.5                         (timeout)
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

POLE_R        = 8.0   # Matches standard futsal map pole physics
POLE_BCOEF    = 0.1
POLE_IMASS    = 0.0   # Immovable

OUTER_PAD     = 35.0  # players can go this far outside field lines

# ─────────────────────────────────────────────────────────────────────────────
# Observation normalisation constants (matches agent-api.js)
# ─────────────────────────────────────────────────────────────────────────────
NORM      = 800.0
MAX_SPEED = 10.0
DIAG      = math.sqrt(NORM ** 2 + NORM ** 2)   # ≈ 1131.4
N_TM      = 4   # max teammate slots
N_OPP     = 5   # max opponent slots
OBS_DIM   = 4 + 4 + 4 + 11 + 2 + N_TM * 9 + N_OPP * 9   # = 106

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

# A1 Dual-Episode Curriculum
# - Precision episode:  small goal (0.3–0.6×), no opponent
# - Opponent episode:   goal shrinks 1.4× → 0.6× over A1_OPP_CURRICULUM_STEPS
A1_PRECISION_RATIO      = 0.20           # 20% episodes are precision (no opponent)
A1_OPP_CURRICULUM_STEPS = 2_000_000      # steps before goal fully narrows
A1_OPP_GOAL_START       = 1.4           # goal scale at A1 t=0
A1_OPP_GOAL_END         = 0.6           # goal scale at t >= A1_OPP_CURRICULUM_STEPS


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
        else:
            ep_secs = 90 # 1.5 minutes
            self.scores = [0, 0]
            
        self.max_steps = ep_secs * PHYSICS_HZ // FRAME_SKIP
        self.step_count = 0

        # Randomize team each episode → shared policy via x-flip
        self.team_id      = int(self._rng.integers(1, 3))   # 1 or 2
        self._flip        = 1.0 if self.team_id == 1 else -1.0
        self._attack_sign = 1   if self.team_id == 1 else -1

        self._reset_positions()
        
        # Init dense-reward tracking
        a = self.agents[0]
        self._prev_dist_to_ball = math.hypot(a.x - self.ball.x, a.y - self.ball.y)
        
        goal_x = self.HW * self._attack_sign
        self._prev_ball_dist_to_goal = math.hypot(goal_x - self.ball.x, self.goal_center_y - self.ball.y)
        self._prev_ball_x       = self.ball.x
        self._prev_ball_speed   = math.hypot(self.ball.xs, self.ball.ys)
        self.last_touch = None

        return self._get_obs(), {}

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
            
        else: # A1
            self.HH = float(preset[1])
            self.HW = float(preset[0])
            self.goal_center_y = 0.0

            # ── Choose episode type ───────────────────────────────────────────
            if self.forced_opponent_type is not None:
                # External override: always use this opponent type (e.g. 'Human' from eval_render)
                self.episode_type    = 'opponent'
                self.opponent_type   = self.forced_opponent_type
                self.opponent_policy = None
                t        = min(self.total_timesteps_elapsed, A1_OPP_CURRICULUM_STEPS)
                progress = t / A1_OPP_CURRICULUM_STEPS
                scale    = A1_OPP_GOAL_START + (A1_OPP_GOAL_END - A1_OPP_GOAL_START) * progress
                self.goal_y = float(preset[2]) * scale
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
            self.ball = Disc(0.0, 0.0, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
            self.agents = []
            if self.episode_type == 'precision':
                # No opponent — just agent
                pos = self._safe_spawn()
                self.agents.append(Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
            else:
                # Red on left, Blue on right
                if self.team_id == 1:
                    self.agents.append(Disc(-self.HW * 0.5, 0.0, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Agent (RED)
                    self.agents.append(Disc( self.HW * 0.5, 0.0, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Opp (BLUE)
                else:
                    self.agents.append(Disc( self.HW * 0.5, 0.0, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Agent (BLUE)
                    self.agents.append(Disc(-self.HW * 0.5, 0.0, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)) # Opp (RED)

    def _safe_spawn(self, max_tries: int = 80):
        """Return (x, y) not overlapping ball or already-placed agents."""
        min_r_ball = PLYR_R + BALL_R + 8.0
        min_r_plyr = PLYR_R * 2 + 8.0
        HW, HH = self.HW, self.HH

        for _ in range(max_tries):
            x = float(self._rng.uniform(-HW * 0.85 + PLYR_R, HW * 0.85 - PLYR_R))
            y = float(self._rng.uniform(-HH * 0.85 + PLYR_R, HH * 0.85 - PLYR_R))

            if math.hypot(x - self.ball.x, y - self.ball.y) < min_r_ball:
                continue

            ok = all(
                math.hypot(x - d.x, y - d.y) >= min_r_plyr
                for d in self.agents
            )
            if ok:
                return (x, y)

        # Fallback: quarter-field opposite to ball
        sx = -1.0 if self.ball.x > 0 else 1.0
        return (sx * HW * 0.5, 0.0)



    # ─── step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        assert self.ball is not None, "Call reset() before step()"

        dir_idx = int(action[0])
        kick    = int(action[1])
        dx, dy  = DIR_MAP[dir_idx]
        
        agent_actions = [(dx, dy, kick)]
        
        # In A1, we have an opponent. Determine their action.
        if self.phase == 'A1' and len(self.agents) > 1:
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
                opp_action, _ = self.opponent_policy.predict(opp_obs, deterministic=True)
                opp_dir_idx = int(opp_action[0])
                opp_kick = int(opp_action[1])
                opp_dx, opp_dy = DIR_MAP[opp_dir_idx]
                agent_actions.append((opp_dx, opp_dy, opp_kick))
            elif self.opponent_type == 'Human':
                agent_actions.append(self.human_opponent_action)
            else:
                agent_actions.append((0.0, 0.0, 0)) # Default/None

        # Run FRAME_SKIP ticks
        goal_result = 0   # 0=none, 1=left(own for red), 2=right(scored for red)
        
        # Track touches for the whole step
        touch_events = []
        
        for _ in range(FRAME_SKIP):
            self.last_touch_this_tick = None
            result = self._tick(agent_actions)
            if self.last_touch_this_tick is not None:
                touch_events.append(self.last_touch_this_tick)
                
            if result != 0:
                goal_result = result
                break

        self.step_count += 1

        # Snapshot after move
        a = self.agents[0]
        cur_dist     = math.hypot(a.x - self.ball.x, a.y - self.ball.y)
        cur_ball_spd = math.hypot(self.ball.xs, self.ball.ys)

        # ── Reward ────────────────────────────────────────────────────────────
        reward     = 0.0
        terminated = False
        truncated = False
        atk = self._attack_sign

        if self.phase == 'A0':
            if goal_result == 2:
                # Scored into right goal ✅
                reward = 5.0
                terminated = True
            else:
                reward -= 0.003 # Dense punishment per step
                
            if not terminated and self.step_count >= self.max_steps:
                truncated = True

        else: # A1 Phase
            # Goal logic
            if goal_result == 2:
                # Agent scored
                self.scores[self.team_id - 1] += 1
                reward += 5.0
                self._reset_positions()
            elif goal_result == 1:
                # Opponent scored
                opp_id = 2 if self.team_id == 1 else 1
                self.scores[opp_id - 1] += 1
                reward -= 5.0
                self._reset_positions()

            # Check if someone won
            if self.scores[0] >= 3 or self.scores[1] >= 3:
                terminated = True
                
            if not terminated:
                # Step penalty: higher when opponent has possession
                if self.last_touch == 'O':
                    reward -= 0.001   # opponent controls ball
                else:
                    reward -= 0.0006  # agent or no one touched last
            
            if not terminated and self.step_count >= self.max_steps:
                truncated = True
                
            # Touch shaping logic
            for t in touch_events:
                if self.last_touch == 'A' and t == 'O':
                    # Ball went from Agent to Opponent -> Lost possession
                    reward -= 0.5
                elif self.last_touch == 'O' and t == 'A':
                    # Ball went from Opponent to Agent -> Regained possession
                    reward += 0.5
                self.last_touch = t

        # Update previous tracking
        self._prev_dist_to_ball = math.hypot(a.x - self.ball.x, a.y - self.ball.y)
        goal_x = self.HW * atk
        self._prev_ball_dist_to_goal = math.hypot(goal_x - self.ball.x, self.goal_center_y - self.ball.y)
        self._prev_ball_speed = math.hypot(self.ball.xs, self.ball.ys)

        return self._get_obs(), float(reward), terminated, truncated, {}

    # ─── Physics ──────────────────────────────────────────────────────────────
    def _tick(self, agent_actions) -> int:
        """
        One physics tick. Takes a list of (dx, dy, kick) for each agent.
        Returns 0=normal | 1=left-goal | 2=right-goal.
        """
        ball   = self.ball
        agents = self.agents
        HW, HH = self.HW, self.HH

        # 1. Kick (before movement)
        for i, ag in enumerate(agents):
            if i < len(agent_actions):
                dx, dy, kick = agent_actions[i]
                if kick:
                    dx_b = ball.x - ag.x
                    dy_b = ball.y - ag.y
                    dist = math.hypot(dx_b, dy_b)
                    if dist > 0 and dist - ag.radius - ball.radius < KICK_RANGE:
                        nx, ny = dx_b / dist, dy_b / dist
                        ball.xs += nx * KICK_STR
                        ball.ys += ny * KICK_STR

        # 2. Acceleration
        for i, ag in enumerate(agents):
            if i < len(agent_actions):
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
        all_discs = [ball] + agents
        n = len(all_discs)
        
        # Track touches in this tick
        touched_agent = False
        touched_opp = False
        
        for i in range(n):
            for j in range(i + 1, n):
                collided = _resolve_dd(all_discs[i], all_discs[j])
                if collided:
                    # Check if player hit the ball
                    if (i == 0 and j > 0) or (j == 0 and i > 0):
                        idx = max(i, j) - 1 # 0 for agent, 1 for opponent
                        if idx == 0:
                            touched_agent = True
                        elif idx == 1:
                            touched_opp = True

        if touched_agent and touched_opp:
            # Both touched in same tick, ambiguous but let's favor the last one or None
            # Standard is just to update who hit it. Let's say agent wins ties.
            self.last_touch_this_tick = 'A'
        elif touched_agent:
            self.last_touch_this_tick = 'A'
        elif touched_opp:
            self.last_touch_this_tick = 'O'
        else:
            self.last_touch_this_tick = None

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

        # Wall collisions — Players (outer boundary, 35px outside field)
        oxW = HW + OUTER_PAD
        oxH = HH + OUTER_PAD
        for ag in agents:
            if ag.x - ag.radius < -oxW:
                ag.x = -oxW + ag.radius; ag.xs =  abs(ag.xs) * 0.3
            if ag.x + ag.radius > oxW:
                ag.x =  oxW - ag.radius; ag.xs = -abs(ag.xs) * 0.3
            if ag.y - ag.radius < -oxH:
                ag.y = -oxH + ag.radius; ag.ys =  abs(ag.ys) * 0.3
            if ag.y + ag.radius > oxH:
                ag.y =  oxH - ag.radius; ag.ys = -abs(ag.ys) * 0.3

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
        obs[i] = 0.0 if self.team_id == 1 else 1.0;    i += 1  # 0=RED, 1=BLUE

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

        # Sections 5 & 6 — Teammates × 4 + Opponents × 5, all zeros in A0
        # (i advances to 106 through the zero-filled array)

        assert i == 25, f"Obs pointer mismatch before padding: {i}"
        # Remaining [25..105] stay 0 (no teammates / opponents in A0/A1 for now since A1 is just 1v1 and the python env isn't using sections 5 and 6 yet)
        # TODO: A1 should technically populate Section 6 (Opponent) with the 1 opponent's data if we want to mimic JS

        return obs

    def _get_obs_for_opponent(self) -> np.ndarray:
        # Swap team context and return obs to opponent (for self play evaluation inside step)
        original_team = self.team_id
        # Swap explicitly
        old_flip = self._flip
        old_attack = self._attack_sign
        
        self.team_id = 2 if self.team_id == 1 else 1
        self._flip = 1.0 if self.team_id == 1 else -1.0
        self._attack_sign = 1 if self.team_id == 1 else -1
        
        # We also need to swap agents[0] and agents[1] for the perspective of `_get_obs`
        self.agents[0], self.agents[1] = self.agents[1], self.agents[0]
        
        obs_opp = self._get_obs()
        
        # Revert
        self.agents[0], self.agents[1] = self.agents[1], self.agents[0]
        self.team_id = original_team
        self._flip = old_flip
        self._attack_sign = old_attack
        return obs_opp

    def _get_follower_action(self):
        """Chase ball and kick toward opponent goal.
        Anti-own-goal: simulate ball trajectory after kick (ray-goal intersection).
        - If kick would enter own goal → suppress + reposition laterally.
        - Otherwise → kick freely regardless of distance.
        """
        ag = self.agents[1]
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

    def _get_attacker_action(self):
        """Aggressively tries to score into agent's goal.
        Positions behind the ball (own-goal side) so kicks go toward agent's goal.
        Has anti-own-goal ray test: never kicks into own goal.
        """
        ag = self.agents[1]
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


    def _get_defender_action(self):
        # Stay near goal, intercept if ball comes near
        ag = self.agents[1]
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

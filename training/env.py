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
OBS_DIM = 4 + 4 + 4 + 11 + 2 + N_TM * 9 + N_OPP * 9 + 3 + 3 + 1 + 5 + 4 + N_TM * 4 + N_OPP * 4  # = 158

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

    def __init__(self, n_agents: int = 1, seed: Optional[int] = None):
        super().__init__()

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
        self.possession_history = deque(maxlen=15)  # 0.25s at 60Hz physics (FRAME_SKIP is used, but physics ticks are 60Hz. step() represents frame_skip ticks. Wait, history should be in terms of real time. If step is 6 ticks, 15 ticks = 2.5 steps. We'll store (time_sec, team_id) and clean old ones.)
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

    # ─── reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ep_secs = 90 # 1.5 minutes
        self.max_steps = ep_secs * PHYSICS_HZ // self.frame_skip
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
        self.possession_history.clear()
        self.investment_sequence.clear()
        self.opp_possession_time = 0.0
        self.dribble_start_time = 0.0
        self.self_pass_shooter = None
        self.self_pass_active = False
        self.real_pass_active = False

        return self._get_obs(), {}

    def _reset_positions(self):
        size_class = '1v1'
        cands = [p for p in MAP_PRESETS if p[3] == size_class] or MAP_PRESETS
        preset = cands[int(self._rng.integers(0, len(cands)))]
        
        self.HH = float(preset[1])
        self.HW = float(preset[0])
        self.goal_y = float(preset[2])
        self.goal_center_y = 0.0
        
        # Determine opponent type (keep only Random, Pazzo, Wanderer, and maybe Static)
        # We will randomly pick from the "4 random bots". Let's assume Static is the 4th since it has no logic.
        # If user meant something else, we can adjust.
        r = self._rng.random()
        if r < 0.25:
            self.opponent_type = 'Random'
        elif r < 0.5:
            self.opponent_type = 'Pazzo'
        elif r < 0.75:
            self.opponent_type = 'Wanderer'
        else:
            self.opponent_type = 'Static'
            
        self.episode_type = 'opponent'
        self.opponent_policy = None

        # Positions
        self.agents = []
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
            bxs *= self.ball.bCoef
            bys *= self.ball.bCoef
            
            for ag in self.agents:
                dist = math.hypot(ag.x - bx, ag.y - by)
                # Player reach: radius + ball_radius + max_speed * t
                # max_speed is roughly 3.0
                if dist <= (ag.radius + self.ball.radius + 3.0 * t):
                    return ag
        return None

    def step(self, action):
        assert self.ball is not None, "Call reset() before step()"

        dir_idx = int(action[0])
        kick    = int(action[1])
        dx, dy  = DIR_MAP[dir_idx]
        
        agent_actions = [(dx * self._flip, dy, kick)]
        
        # Determine opponent action
        if self.phase in ('A1', 'A1.2', 'A0.1') and len(self.agents) > 1:
            if self.opponent_type == 'Defender':
                agent_actions.append(self._get_defender_action())
            elif self.opponent_type == 'Attacker':
                agent_actions.append(self._get_attacker_action())
            elif self.opponent_type == 'Hybrid':
                if self.last_touch == 'A':
                    self._hybrid_mode = 'defender'
                elif self.last_touch == 'O':
                    self._hybrid_mode = 'follower'
                if self._hybrid_mode == 'defender':
                    agent_actions.append(self._get_defender_action())
                else:
                    agent_actions.append(self._get_follower_action())
            elif self.opponent_type == 'Trained' and self.opponent_policy is not None:
                opp_obs = self._get_obs_for_opponent()
                opp_action, _ = self.opponent_policy.predict(opp_obs, deterministic=False)
                opp_dx, opp_dy = DIR_MAP[int(opp_action[0])]
                agent_actions.append((opp_dx * -self._flip, opp_dy, int(opp_action[1])))
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
                agent_actions.append((0.0, 0.0, 0))

        goal_result = 0
        touch_events = []
        
        for _ in range(self.frame_skip):
            self.last_touch_player_this_tick = None
            self.last_touch_team_this_tick = None
            result = self._tick(agent_actions)
            if self.last_touch_player_this_tick is not None:
                touch_events.append((self.last_touch_player_this_tick, self.last_touch_team_this_tick))
                
            if result != 0:
                goal_result = result
                break

        # ── MARL Possession & Investment Sequence Logic ───────────────────────
        self.step_count += 1
        time_now = self.step_count * (self.frame_skip / 60.0)
        
        # Clean up old possession history (> 0.25s ago)
        while self.possession_history and (time_now - self.possession_history[0][0]) > 0.25:
            self.possession_history.popleft()
            
        possession_0_25s = self.possession_history[0][1] if self.possession_history else None

        opp_id = 2 if self.team_id == 1 else 1

        turnover_penalty = 0.0
        for pid, tid in touch_events:
            # Detect turnover: Ball was ours, now touched by opponent
            if self.last_touch_team == self.team_id and tid == opp_id:
                turnover_penalty -= 0.1
                
            self.last_touch = 'A' if pid == 0 else 'O'
            self.last_touch_team = tid
            self.possession_history.append((time_now, tid))
            
            # Reset dribble timer if different player touched
            if hasattr(self, '_last_touch_pid') and self._last_touch_pid != pid:
                self.dribble_start_time = time_now
            elif not hasattr(self, '_last_touch_pid'):
                self.dribble_start_time = time_now
            self._last_touch_pid = pid

            # Investment Sequence Tracking
            if tid == opp_id:
                # If opponent touched, do not immediately clear sequence unless they hold for 2s.
                # However, if self-pass active and touches opponent, remove self-pass limit
                self.self_pass_active = False
                self.real_pass_active = False
            else:
                # Teammate (or self) touched
                self.self_pass_active = False
                self.real_pass_active = False # Hit a teammate
                if pid == 0:
                    # Agent (Tôi) touched -> reset sequence
                    self.investment_sequence = [pid]
                else:
                    # Teammate touched
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

        # Anti-self pass detection on KICK using Greedy Prediction
        if kick and self.last_touch_team == self.team_id:
            # Predict the receiver of the ball
            receiver = self._predict_greedy_receiver()
            # If the predicted receiver is the agent itself (agents[0]), it's a self pass
            if receiver is self.agents[0]:
                self.self_pass_active = True
                self.real_pass_active = False
            else:
                self.self_pass_active = False
                self.real_pass_active = True
        self.real_pass_active = False

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
            # Zone 1 (Sân nhà)
            own_goal_x = -self.HW * atk
            cur_dist = math.hypot(bx - own_goal_x, by)
            prev_dist = math.hypot(px - own_goal_x, py)
            delta_dist_to_goal = cur_dist - prev_dist
        elif adv_x >= self.HW - zone_width:
            # Zone 3 (Sân khách)
            opp_goal_x = self.HW * atk
            cur_dist = math.hypot(bx - opp_goal_x, by)
            prev_dist = math.hypot(px - opp_goal_x, py)
            delta_dist_to_goal = prev_dist - cur_dist
        else:
            # Zone 2 (Giữa sân)
            delta_dist_to_goal = (bx - px) * atk
        
        # Ball Movement Reward Rules
        # + if (current_pos == self or pos_0_25 == self) and moving forward
        # - if (current_pos == opp or pos_0_25 == self) and moving backward
        has_possession_reward = (self.last_touch_team == self.team_id) or (possession_0_25s == self.team_id)
        has_possession_penalty = (self.last_touch_team == opp_id) or (possession_0_25s == self.team_id)
        
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
        
        # Exception: Pass back to teammate -> 1/3 penalty
        is_pass_back = (possession_0_25s == self.team_id and self.last_touch_team == self.team_id)
        
        # Exception: Opp pass back to opp -> no reward
        is_opp_pass_back = (possession_0_25s == opp_id and self.last_touch_team == opp_id)
        
        dribble_duration = time_now - self.dribble_start_time
        is_dribbling = (dribble_duration > 0.0) # NO DELAY! threshold is 0.0 seconds
        
        if delta_dist_to_goal > 0: # Ball advanced (good)
            if has_possession_reward and not is_opp_pass_back:
                # Calculate multiplier
                mult = 1.0
                if self.real_pass_active:
                    mult = 1.0
                elif self.self_pass_active:
                    mult = 0.333
                elif is_dribbling and not kick:
                    mult = 0.333
                
                # Multiply by invest_share (Reward is shared)
                reward += ADVANCE_REWARD * delta_dist_to_goal * mult * invest_share
                
        elif delta_dist_to_goal < 0: # Ball moved back (bad)
            if has_possession_penalty:
                mult = 1.0
                if is_pass_back:
                    mult = 0.333
                
                # Penalty is cushioned by invest_share if in sequence AND it's our possession (pass back)
                # If opponent has possession, EVERYONE gets 100% penalty (ai cũng nhận hình phạt như nhau)
                if self.last_touch_team == opp_id:
                    penalty_share = 1.0
                else:
                    penalty_share = invest_share if in_sequence else 1.0
                    
                reward -= BACKWARD_PENALTY * abs(delta_dist_to_goal) * mult * penalty_share

        # Apply Instant Turnover Penalty
        if turnover_penalty < 0.0:
            if in_sequence:
                # Investors take % penalty
                reward += turnover_penalty * invest_share
            else:
                # Non-investors take 100% penalty
                reward += turnover_penalty * 1.0

        # Goal Logic
        goal_scored = False
        if goal_result == 2:
            self.scores[self.team_id - 1] += 1
            base_reward = 12.0
            
            # Investor Reward (Agent only gets the base reward if they scored directly, but if a teammate scored, 
            # agent gets investor share if in sequence. Since this is single agent control for now, if sequence has agent, they get share.)
            # If agent is the scorer (pid == 0 in sequence[-1]):
            if self.investment_sequence and self.investment_sequence[-1] == 0:
                reward += base_reward
            elif 0 in self.investment_sequence:
                idx = self.investment_sequence.index(0)
                # Investor formula: 30% * (1/2)^(N-1)
                N = len(self.investment_sequence) - idx
                invest_share = 0.3 * (0.5 ** (N - 1))
                reward += base_reward * invest_share
            else:
                reward += base_reward # If agent wasn't involved at all, still give full reward? Or maybe just base?

            goal_scored = True
            terminated = True
            
        elif goal_result == 1:
            self.scores[opp_id - 1] += 1
            reward -= 10.0
            goal_scored = True
            terminated = True

        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        # Update previous tracking
        self._prev_dist_to_ball = math.hypot(self.agents[0].x - self.ball.x, self.agents[0].y - self.ball.y)
        self._prev_ball_dist_to_goal = cur_ball_dist_to_goal
        self._prev_ball_x       = self.ball.x
        self._prev_ball_y       = self.ball.y
        self._prev_ball_speed   = math.hypot(self.ball.xs, self.ball.ys)

        info = {
            "marl/sequence_len": len(self.investment_sequence),
            "marl/dribble_duration": float(dribble_duration),
            "marl/self_pass": int(self.self_pass_active),
            "marl/opp_pos_time": float(self.opp_possession_time),
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
        shuffled_agents = list(agents)
        self._rng.shuffle(shuffled_agents)
        all_discs = [ball] + shuffled_agents
        n = len(all_discs)
        
        self.last_touch_player_this_tick = None
        self.last_touch_team_this_tick = None
        
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
                            self.last_touch_player_this_tick = orig_idx
                            self.last_touch_team_this_tick = self.team_id if orig_idx == 0 else (2 if self.team_id == 1 else 1)
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
    def _get_obs(self) -> np.ndarray:
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
        obs[i] = 0.0;    i += 1  # 0=RED, 1=BLUE

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

        obs[i] = (self.HW - flip * bx) / NORM;       i += 1
        obs[i] = (opp_post_y - by) / NORM;           i += 1
        obs[i] = (-self.HW - flip * bx) / NORM;      i += 1
        obs[i] = (own_post_y - by) / NORM;           i += 1

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
        
        # Possession 0.25s ago (3 dims)
        pos_0_25 = self.possession_history[0][1] if self.possession_history else None
        if pos_0_25 is None:
            obs[i:i+3] = [1, 0, 0]
        elif pos_0_25 == self.team_id:
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
            theta = math.atan2(dy, dx)
            alpha = math.asin(R / d)
            v1_x = L * math.cos(theta + alpha)
            v1_y = L * math.sin(theta + alpha)
            v2_x = L * math.cos(theta - alpha)
            v2_y = L * math.sin(theta - alpha)
            return flip * v1_x / NORM, v1_y / NORM, flip * v2_x / NORM, v2_y / NORM

        # Main agent tangent vectors (4 dims)
        t_mx1, t_my1, t_mx2, t_my2 = _get_tangent_vectors(mx, my)
        obs[i:i+4] = [t_mx1, t_my1, t_mx2, t_my2]
        i += 4

        # Sections 5 & 6 — Teammates × 4 + Opponents × 5
        # The base code has 9 dims per TM and 9 dims per OPP.
        # We will append the 4 tangent dims directly after them? No, we should place them systematically.
        # The easiest is to just leave empty padding or adjust the index.
        # N_TM * 9 (old) + N_TM * 4 (new)
        # Wait, the prompt said "boi player, bo sung vector tu bong de tiep diem..."
        # We'll just write the N_TM and N_OPP loops.
        # In this 1v1 setup, no teammates.
        
        # Teammates (each takes 14 dims: 9 base + 4 tangent + 1 share)
        for tm_i in range(N_TM):
            idx = i + tm_i * 14
            # Agent indexes are 0 (main), 1 to N_TM (teammates)
            tm_agent_idx = tm_i + 1
            share = 0.0
            if self.investment_sequence:
                if self.investment_sequence[-1] == tm_agent_idx:
                    share = 1.0  # Holder
                elif tm_agent_idx in self.investment_sequence:
                    seq_idx = self.investment_sequence.index(tm_agent_idx)
                    passes_away = len(self.investment_sequence) - 1 - seq_idx
                    share = 0.3 * (0.5 ** (passes_away - 1))
            
            # Since we have no teammates in 1v1 currently, we just leave the base 14 dims as 0
            # If we had teammates, we would populate obs[idx:idx+14] here.
            obs[idx + 13] = share  # the 14th dimension
        i += N_TM * 14

        if len(self.agents) > 1:
            opp = self.agents[1]
            idx = i
            
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

        i += N_OPP * 13

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


    def _get_static_action(self):
        """StaticModel: Always stays still"""
        return (0.0, 0.0, 0)

    def _get_random_action(self):
        """RandomModel: Randomly moves and kicks"""
        dir_idx = int(self._rng.integers(0, 9))
        kick = int(self._rng.integers(0, 2))
        dx, dy = DIR_MAP[dir_idx]
        return (dx, dy, kick)

    def _get_pazzo_action(self):
        """PazzoModel: Moves to a random point, changes every rand[100,150] steps. Jitters if ball is near center."""
        ag = self.agents[1]
        b = self.ball

        if not hasattr(self, '_pazzo_steps'):
            self._pazzo_steps = 0
            self._pazzo_target = None
            self._pazzo_interval = int(self._rng.integers(100, 151))

        self._pazzo_steps += 1

        if self._pazzo_target is None or self._pazzo_steps > self._pazzo_interval:
            self._pazzo_target = (
                self._rng.uniform(-self.HW, self.HW),
                self._rng.uniform(-self.HH, self.HH)
            )
            self._pazzo_steps = 0
            self._pazzo_interval = int(self._rng.integers(100, 151))

        # Nếu bóng ở trung tâm: jitter ngẫu nhiên ngắm mục tiêu
        if abs(b.x) < 5.0 and abs(b.y) < 5.0 and math.hypot(b.xs, b.ys) < 0.5:
            target_x = self._rng.uniform(-150.0, 150.0)
            target_y = self._rng.uniform(-150.0, 150.0)
        else:
            target_x, target_y = self._pazzo_target

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

    def _get_wanderer_action(self):
        """WandererBot: every rand[3,12] steps picks a point within radius 30 of the ball
        and moves toward it. Kicks ~30% when in range."""
        ag = self.agents[1]
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

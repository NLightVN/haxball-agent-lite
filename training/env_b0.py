"""
HaxballB0Env — B0 (Defend Bot) Training Environment
=====================================================
Task  : B0 defends goal_B0. Opponent A0 (frozen, follower-bot or trained PPO)
        attacks from the opposite half.
        Episode ends when A0 scores into goal_B0 OR 15 seconds (150 steps) elapse.

Field : Fixed 1v1 preset (368 × 171, goal_y=64).  Team side randomised each
        episode so B0 learns a shared policy via x-flip.

Action space : MultiDiscrete([9, 2])  — same as A0/A1

Observation  : 106-dim float32 (same layout as env.py, goal_B0 always on the
               left from B0's flipped perspective)

Reward (per step / per event):
    +2/300 per step — survival reward each step (~+0.00667/step)
    -0.01/step      — B0 inside own goal > 5 consecutive steps
    -5.0            — A0 scores into goal_B0 (A0+ball reset, episode continues)
    no reward       — B0 scores into goal_A0 → reset A0+ball, continue
    Episode ends only on timeout (300 steps).
"""

import math
from typing import Optional, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ─────────────────────────────────────────────────────────────────────────────
# Physics constants (must match env.py / test_index.html exactly)
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
KICK_RANGE    = 4.0

POLE_R        = 8.0
POLE_BCOEF    = 0.1
POLE_IMASS    = 0.0

OUTER_PAD     = 35.0

# Wall bounce coefficient (futsal walls are hard)
WALL_BCOEF    = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Observation normalisation constants
# ─────────────────────────────────────────────────────────────────────────────
NORM          = 800.0
MAX_SPEED     = 10.0
DIAG          = math.sqrt(NORM ** 2 + NORM ** 2)
N_TM          = 4
N_OPP         = 5
OBS_DIM       = 4 + 4 + 4 + 11 + 2 + N_TM * 9 + N_OPP * 9  # = 106

# ─────────────────────────────────────────────────────────────────────────────
# Training constants
# ─────────────────────────────────────────────────────────────────────────────
FRAME_SKIP        = 6
PHYSICS_HZ        = 60
MAX_STEPS         = 300  # 300 steps = 30 s @ 60Hz / FRAME_SKIP=6

MAX_STEPS_ALL_MODES = 6000  # for time-normalisation in obs

# Fixed map preset (1v1 futsal)
MAP_HW   = 368.0
MAP_HH   = 171.0
MAP_GOAL = 64.0

# Trajectory prediction limit (steps to simulate)
TRAJ_MAX_STEPS = 300

# Player terminal velocity (≈ PLYR_ACC / (1 - PLYR_DAMP))
PLYR_MAX_SPEED = PLYR_ACC / (1.0 - PLYR_DAMP)  # ≈ 2.75 px/tick

# B0 intercept radius (standing "at" the intercept point)
INTERCEPT_RADIUS = PLYR_R + BALL_R + 10.0

# ─────────────────────────────────────────────────────────────────────────────
# Direction map
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
    """Mutable physics disc."""
    __slots__ = ['x', 'y', 'xs', 'ys', 'radius', 'imass', 'bcoef', 'damp']

    def __init__(self, x, y, xs, ys, radius, imass, bcoef, damp):
        self.x = float(x); self.y = float(y)
        self.xs = float(xs); self.ys = float(ys)
        self.radius = float(radius)
        self.imass  = float(imass)
        self.bcoef  = float(bcoef)
        self.damp   = float(damp)

    def copy(self):
        return Disc(self.x, self.y, self.xs, self.ys,
                    self.radius, self.imass, self.bcoef, self.damp)


# ─────────────────────────────────────────────────────────────────────────────
# Ball trajectory prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_ball_trajectory(
    ball: Disc,
    HW: float, HH: float,
    goal_center_y: float, goal_y: float,
    goal_sign: float,        # -1 = goal_B0 on left (-HW), +1 = goal_B0 on right (+HW)
    max_steps: int = TRAJ_MAX_STEPS,
) -> Tuple[List[Tuple[float, float]], bool]:
    """
    Simulate ball trajectory (no player interaction).

    Returns
    -------
    points        : list of (x, y) sampled each physics tick
    into_goal_B0  : True if ball eventually enters goal_B0
    """
    # Work on a copy
    bx, by = ball.x, ball.y
    bxs, bys = ball.xs, ball.ys
    bcoef = ball.bcoef

    goal_B0_x  = goal_sign * HW          # e.g. -HW if B0 on left
    goal_A0_x  = -goal_sign * HW         # opposite side

    goal_top = goal_center_y + goal_y
    goal_bot = goal_center_y - goal_y

    points: List[Tuple[float, float]] = []
    into_goal_B0 = False

    for _ in range(max_steps):
        # Check if ball is slow enough to stop tracing
        speed = math.hypot(bxs, bys)
        if speed < 0.05:
            break

        bx += bxs
        by += bys

        # Wall: top/bottom
        if by - BALL_R < -HH:
            by = -HH + BALL_R
            bys = -bys * bcoef
        elif by + BALL_R > HH:
            by = HH - BALL_R
            bys = -bys * bcoef

        # Wall/goal: left side
        if bx - BALL_R < goal_B0_x * math.copysign(1, goal_sign):
            # Check which x-limit we mean
            # goal_sign = -1 → goal_B0 at -HW, so check bx-r < -HW
            # goal_sign = +1 → goal_B0 at +HW, so check bx+r > +HW
            pass  # handled below in unified check

        # Unified left/right checks
        hit_left  = (bx - BALL_R < -HW)
        hit_right = (bx + BALL_R >  HW)

        if hit_left:
            in_goal = goal_bot < by < goal_top
            if in_goal:
                # Ball enters left goal
                if goal_sign == -1:
                    into_goal_B0 = True
                points.append((bx, by))
                return points, into_goal_B0
            else:
                bx = -HW + BALL_R
                bxs = -bxs * bcoef

        elif hit_right:
            in_goal = goal_bot < by < goal_top
            if in_goal:
                # Ball enters right goal
                if goal_sign == +1:
                    into_goal_B0 = True
                points.append((bx, by))
                return points, into_goal_B0
            else:
                bx = HW - BALL_R
                bxs = -bxs * bcoef

        # Damping
        bxs *= BALL_DAMP
        bys *= BALL_DAMP

        points.append((bx, by))

    return points, into_goal_B0


# ─────────────────────────────────────────────────────────────────────────────
# Agent intercept time estimator
# ─────────────────────────────────────────────────────────────────────────────
def steps_to_intercept(
    ax: float, ay: float,
    ball_line: List[Tuple[float, float]],
) -> Tuple[float, int]:
    """
    Estimate the minimum number of physics ticks for an agent at (ax, ay)
    to reach any point on ball_line at that point's arrival time.

    The agent accelerates toward a target every tick.  We approximate the
    distance-to-time conversion using the analytic formula for constant
    acceleration with damping: the agent reaches distance d in roughly
        t ≈ d / (PLYR_ACC / (1-PLYR_DAMP))  ticks  (lower-bound, instant max speed)

    For each ball_line point i (arrives at physics tick i+1), check whether
    the agent can get there by tick i+1.

    Returns (min_arrival_ticks, best_index).
    If no point is reachable, returns (inf, -1).
    """
    if not ball_line:
        return (math.inf, -1)

    best_ticks = math.inf
    best_idx   = -1

    for i, (px, py) in enumerate(ball_line):
        ball_tick = i + 1                          # ball arrives at this tick
        dist = math.hypot(px - ax, py - ay) - PLYR_R - BALL_R
        dist = max(0.0, dist)

        # Time for agent to cover `dist` at max speed (optimistic lower bound)
        agent_ticks = dist / PLYR_MAX_SPEED if PLYR_MAX_SPEED > 0 else math.inf

        if agent_ticks <= ball_tick:
            if ball_tick < best_ticks:
                best_ticks = ball_tick
                best_idx   = i
            break  # first reachable point is the earliest — stop

    return (best_ticks, best_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Collision helper (exact port from env.py)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_dd(da: Disc, db: Disc) -> bool:
    ddx = da.x - db.x; ddy = da.y - db.y
    dist = math.hypot(ddx, ddy)
    r_sum = da.radius + db.radius
    if 0 < dist <= r_sum:
        nx, ny = ddx / dist, ddy / dist
        imass_sum = da.imass + db.imass
        if imass_sum == 0:
            return True
        mf = da.imass / imass_sum
        overlap = r_sum - dist
        da.x += nx * overlap * mf;       da.y += ny * overlap * mf
        db.x -= nx * overlap * (1 - mf); db.y -= ny * overlap * (1 - mf)
        rvn = (da.xs - db.xs) * nx + (da.ys - db.ys) * ny
        if rvn < 0:
            impulse = rvn * (da.bcoef * db.bcoef + 1)
            da.xs -= nx * impulse * mf;       da.ys -= ny * impulse * mf
            db.xs += nx * impulse * (1 - mf); db.ys += ny * impulse * (1 - mf)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
class HaxballB0Env(gym.Env):
    """
    B0 defend-bot environment.

    agents[0] = B0 (the learning agent, defends goal_B0)
    agents[1] = A0 (frozen opponent, attacks goal_B0)

    Team assignment is randomised each episode → shared policy via x-flip.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        a0_model_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self._rng = np.random.default_rng(seed)
        self.a0_model_path   = a0_model_path
        self._a0_policy      = None   # loaded lazily on first reset if path given

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([9, 2])

        # Field (fixed per preset, randomised scale)
        self.HW           = MAP_HW
        self.HH           = MAP_HH
        self.goal_y       = MAP_GOAL
        self.goal_center_y = 0.0

        # Team / side info  (set in reset)
        self.team_id      = 1      # 1=RED, 2=BLUE  (B0's team)
        self._flip        = 1.0    # +1 RED, -1 BLUE
        self._goal_sign   = -1.0   # -1: goal_B0 at -HW (B0 is RED, defends left)
                                   # +1: goal_B0 at +HW (B0 is BLUE, defends right)

        # Physics objects
        self.ball: Optional[Disc]  = None
        self.agents: list[Disc]    = []   # [0]=B0, [1]=A0

        # Step counter
        self.step_count = 0
        self.max_steps  = MAX_STEPS

        # Touch tracking
        self.last_touch: Optional[str] = None    # 'B' B0, 'A' A0
        self.last_touch_this_tick: Optional[str] = None

        # Ball-line state (recomputed each step when needed)
        self._ball_line:          List[Tuple[float, float]] = []
        self._ball_line_into_goal: bool = False
        self._prev_steps_to_intercept: float = math.inf
        self._no_intercept_triggered: bool = False  # tracks if penalty was already accumulating

        # Dense-reward tracking
        self._prev_dist_to_ball = 0.0

        # Goal-camping penalty tracker
        self._steps_in_own_goal: int = 0

        # Timestep counter (for future curriculum hooks)
        self.total_timesteps_elapsed: int = 0

    # ─── reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0

        # Random team assignment each episode
        self.team_id    = int(self._rng.integers(1, 3))   # 1=RED, 2=BLUE
        self._flip      = 1.0 if self.team_id == 1 else -1.0
        # B0 as RED defends left (-HW), B0 as BLUE defends right (+HW)
        self._goal_sign = -1.0 if self.team_id == 1 else 1.0

        # Map scaling: 80%–120% of preset
        scale_h = float(self._rng.uniform(0.8, 1.2))
        scale_w = float(self._rng.uniform(0.8, 1.2))
        self.HH = MAP_HH * scale_h
        self.HW = MAP_HW * scale_w
        if self.HH > self.HW:
            self.HH, self.HW = self.HW, self.HH

        # Goal: 60%–120% of preset
        self.goal_y = float(self._rng.uniform(0.6, 1.2)) * MAP_GOAL
        padding = 10.0
        max_center = max(0.0, self.HH - self.goal_y - padding)
        self.goal_center_y = float(self._rng.uniform(-max_center, max_center))

        self._reset_positions()

        # Init ball-line
        self._recompute_ball_line()
        self._prev_steps_to_intercept, _ = steps_to_intercept(
            self.agents[0].x, self.agents[0].y, self._ball_line
        )
        self._no_intercept_triggered = False
        self.last_touch = None
        self._steps_in_own_goal = 0
        self._prev_dist_to_ball = math.hypot(
            self.agents[0].x - self.ball.x,
            self.agents[0].y - self.ball.y,
        )

        # Lazy-load A0 PPO policy
        if self._a0_policy is None and self.a0_model_path is not None:
            from stable_baselines3 import PPO as _PPO
            self._a0_policy = _PPO.load(self.a0_model_path, device='cpu')

        return self._get_obs(), {}

    def _reset_positions(self):
        """Spawn B0 in defending half, A0 + ball in attacking half."""
        HW, HH = self.HW, self.HH
        goal_sign = self._goal_sign

        # goal_B0 side: x in [goal_sign*HW, 0]  (i.e. the half containing goal_B0)
        # goal_A0 side: x in [0, -goal_sign*HW]
        b0_x_min = goal_sign * HW if goal_sign < 0 else 0.0
        b0_x_max = 0.0             if goal_sign < 0 else goal_sign * HW
        y_min = self.goal_center_y - self.goal_y
        y_max = self.goal_center_y + self.goal_y

        # B0 spawn: within goal belt on defending side
        b0_x = float(self._rng.uniform(
            min(b0_x_min, b0_x_max) + PLYR_R,
            max(b0_x_min, b0_x_max) - PLYR_R,
        ))
        b0_y = float(self._rng.uniform(
            max(-HH + PLYR_R, y_min - 10),
            min( HH - PLYR_R, y_max + 10),
        ))

        # Ball spawn: attacking half (opposite side)
        a0_half_sign = -goal_sign
        ball_x = float(self._rng.uniform(
            BALL_R if a0_half_sign > 0 else -HW + BALL_R,
            HW - BALL_R if a0_half_sign > 0 else -BALL_R,
        ))
        ball_y = float(self._rng.uniform(-HH * 0.7 + BALL_R, HH * 0.7 - BALL_R))

        # A0 spawn: attacking half, not overlapping ball
        for _ in range(60):
            a0_x = float(self._rng.uniform(
                PLYR_R if a0_half_sign > 0 else -HW + PLYR_R,
                HW - PLYR_R if a0_half_sign > 0 else -PLYR_R,
            ))
            a0_y = float(self._rng.uniform(-HH * 0.8 + PLYR_R, HH * 0.8 - PLYR_R))
            if math.hypot(a0_x - ball_x, a0_y - ball_y) >= PLYR_R + BALL_R + 8:
                break

        self.ball = Disc(ball_x, ball_y, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
        self.agents = [
            Disc(b0_x, b0_y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP),  # B0
            Disc(a0_x, a0_y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP),  # A0
        ]

    def _reset_a0_and_ball(self):
        """Reset A0 + ball to attacking half (after B0 own-goals into goal_A0)."""
        HW, HH = self.HW, self.HH
        a0_half_sign = -self._goal_sign

        ball_x = float(self._rng.uniform(
            BALL_R if a0_half_sign > 0 else -HW + BALL_R,
            HW - BALL_R if a0_half_sign > 0 else -BALL_R,
        ))
        ball_y = float(self._rng.uniform(-HH * 0.7 + BALL_R, HH * 0.7 - BALL_R))

        for _ in range(60):
            a0_x = float(self._rng.uniform(
                PLYR_R if a0_half_sign > 0 else -HW + PLYR_R,
                HW - PLYR_R if a0_half_sign > 0 else -PLYR_R,
            ))
            a0_y = float(self._rng.uniform(-HH * 0.8 + PLYR_R, HH * 0.8 - PLYR_R))
            if math.hypot(a0_x - ball_x, a0_y - ball_y) >= PLYR_R + BALL_R + 8:
                break

        self.ball = Disc(ball_x, ball_y, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
        self.agents[1] = Disc(a0_x, a0_y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)
        self.last_touch = None
        self._recompute_ball_line()

    # ─── step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        assert self.ball is not None, "Call reset() first"

        dir_idx = int(action[0])
        kick    = int(action[1])
        dx, dy  = DIR_MAP[dir_idx]
        b0_action = (dx, dy, kick)

        # A0 action
        a0_action = self._get_a0_action()
        agent_actions = [b0_action, a0_action]

        # ── Run FRAME_SKIP ticks ──────────────────────────────────────────────
        goal_result   = 0    # 0=none, 1=goal_B0 scored into, 2=goal_A0 scored into
        touch_events  = []

        for _ in range(FRAME_SKIP):
            self.last_touch_this_tick = None
            result = self._tick(agent_actions)
            if self.last_touch_this_tick is not None:
                touch_events.append(self.last_touch_this_tick)
            if result != 0:
                goal_result = result
                break

        self.step_count += 1

        # ── Recompute ball-line after physics ─────────────────────────────────
        prev_into_goal = self._ball_line_into_goal
        prev_ball_line = self._ball_line
        prev_steps_ict = self._prev_steps_to_intercept

        self._recompute_ball_line()
        ball_line_changed = self._ball_line_changed(prev_ball_line, prev_into_goal)

        cur_steps_ict, best_idx = steps_to_intercept(
            self.agents[0].x, self.agents[0].y, self._ball_line
        )

        # ── Reward ──────────────────────────────────────────────────────────────
        reward     = 0.0
        terminated = False
        truncated  = False

        # 1. Update possession tracking (no reward, just state)
        for t in touch_events:
            self.last_touch = t

        # 2. Per-step survival reward (×3 when B0 has possession)
        survival = 2.0 / 300.0
        if self.last_touch == 'B':
            survival *= 3.0
        reward += survival

        # 2. Goal results
        if goal_result == 1:
            # A0 scored into goal_B0 → penalise, reset positions, continue episode
            reward -= 5.0
            self._reset_a0_and_ball()
            self._recompute_ball_line()
            self._prev_steps_to_intercept, _ = steps_to_intercept(
                self.agents[0].x, self.agents[0].y, self._ball_line
            )
        elif goal_result == 2:
            # B0 scored into goal_A0 → no reward, reset and continue
            self._reset_a0_and_ball()
            self._recompute_ball_line()
            self._prev_steps_to_intercept, _ = steps_to_intercept(
                self.agents[0].x, self.agents[0].y, self._ball_line
            )

        # 3. Timeout — only termination condition
        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        # 4. Goal-camping penalty: B0 inside own goal > 5 consecutive steps
        b0 = self.agents[0]
        goal_x = self._goal_sign * self.HW
        in_goal = (
            (self._goal_sign < 0 and b0.x < goal_x) or
            (self._goal_sign > 0 and b0.x > goal_x)
        ) and (self.goal_center_y - self.goal_y < b0.y < self.goal_center_y + self.goal_y)

        if in_goal:
            self._steps_in_own_goal += 1
            if self._steps_in_own_goal > 10:
                reward -= 0.01
        else:
            self._steps_in_own_goal = 0

        return self._get_obs(), float(reward), terminated, truncated, {}

    # ─── Ball-line helpers ────────────────────────────────────────────────────
    def _recompute_ball_line(self):
        """Recompute trajectory from current ball state."""
        self._ball_line, self._ball_line_into_goal = predict_ball_trajectory(
            self.ball,
            self.HW, self.HH,
            self.goal_center_y, self.goal_y,
            self._goal_sign,
            max_steps=TRAJ_MAX_STEPS,
        )

    def _ball_line_changed(
        self,
        prev_line: List[Tuple[float, float]],
        prev_into_goal: bool,
    ) -> bool:
        """
        True if ball_line changed meaningfully:
          - into_goal flag flipped, OR
          - last point moved more than 5 px (trajectory bent significantly)
        """
        if prev_into_goal != self._ball_line_into_goal:
            return True
        if not prev_line or not self._ball_line:
            return prev_line != self._ball_line
        # Compare last point
        px, py = prev_line[-1]
        cx, cy = self._ball_line[-1]
        return math.hypot(cx - px, cy - py) > 5.0

    def _dist_ball_to_goal_B0(self) -> float:
        """Distance from current ball position to goal_B0 centre."""
        goal_x = self._goal_sign * self.HW
        return math.hypot(self.ball.x - goal_x, self.ball.y - self.goal_center_y)

    # ─── Physics tick ─────────────────────────────────────────────────────────
    def _tick(self, agent_actions) -> int:
        """
        One physics tick.
        Returns: 0=normal | 1=scored into goal_B0 | 2=scored into goal_A0
        """
        ball   = self.ball
        agents = self.agents
        HW, HH = self.HW, self.HH

        # 1. Kick
        for i, ag in enumerate(agents):
            if i < len(agent_actions):
                dx, dy, kk = agent_actions[i]
                if kk:
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
                dx, dy, kk = agent_actions[i]
                ln = math.hypot(dx, dy)
                acc = PLYR_KICK_ACC if kk else PLYR_ACC
                if ln > 0:
                    ag.xs += (dx / ln) * acc
                    ag.ys += (dy / ln) * acc

        # 3. Move
        ball.x += ball.xs; ball.y += ball.ys
        for ag in agents:
            ag.x += ag.xs; ag.y += ag.ys

        # 4a. Disc-disc collisions
        all_discs = [ball] + agents
        n = len(all_discs)
        touched_b0 = False
        touched_a0 = False

        for i in range(n):
            for j in range(i + 1, n):
                collided = _resolve_dd(all_discs[i], all_discs[j])
                if collided and (i == 0 or j == 0):
                    player_idx = max(i, j) - 1
                    if player_idx == 0:
                        touched_b0 = True
                    elif player_idx == 1:
                        touched_a0 = True

        if touched_b0 and touched_a0:
            self.last_touch_this_tick = 'B'   # B0 wins ties
        elif touched_b0:
            self.last_touch_this_tick = 'B'
        elif touched_a0:
            self.last_touch_this_tick = 'A'
        else:
            self.last_touch_this_tick = None

        # 4b. Goal pole collisions
        poles = [
            Disc( HW, self.goal_center_y + self.goal_y, 0,0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
            Disc( HW, self.goal_center_y - self.goal_y, 0,0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
            Disc(-HW, self.goal_center_y + self.goal_y, 0,0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
            Disc(-HW, self.goal_center_y - self.goal_y, 0,0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
        ]
        for pole in poles:
            for ag in agents:
                _resolve_dd(ag, pole)
            _resolve_dd(ball, pole)

        # 5. Wall collisions — ball
        if ball.y - ball.radius < -HH:
            ball.y = -HH + ball.radius
            ball.ys = -ball.ys * ball.bcoef
        if ball.y + ball.radius > HH:
            ball.y = HH - ball.radius
            ball.ys = -ball.ys * ball.bcoef

        goal_top = self.goal_center_y + self.goal_y
        goal_bot = self.goal_center_y - self.goal_y

        if ball.x - ball.radius < -HW:
            if goal_bot < ball.y < goal_top:
                # Left goal scored
                return 1 if self._goal_sign == -1 else 2
            ball.x = -HW + ball.radius
            ball.xs = -ball.xs * ball.bcoef
        elif ball.x + ball.radius > HW:
            if goal_bot < ball.y < goal_top:
                # Right goal scored
                return 1 if self._goal_sign == 1 else 2
            ball.x = HW - ball.radius
            ball.xs = -ball.xs * ball.bcoef

        # Wall collisions — players
        oxW, oxH = HW + OUTER_PAD, HH + OUTER_PAD
        for ag in agents:
            if ag.x - ag.radius < -oxW: ag.x = -oxW + ag.radius; ag.xs =  abs(ag.xs) * 0.3
            if ag.x + ag.radius >  oxW: ag.x =  oxW - ag.radius; ag.xs = -abs(ag.xs) * 0.3
            if ag.y - ag.radius < -oxH: ag.y = -oxH + ag.radius; ag.ys =  abs(ag.ys) * 0.3
            if ag.y + ag.radius >  oxH: ag.y =  oxH - ag.radius; ag.ys = -abs(ag.ys) * 0.3

        # 6. Damping
        ball.xs *= ball.damp; ball.ys *= ball.damp
        for ag in agents:
            ag.xs *= ag.damp; ag.ys *= ag.damp

        return 0

    # ─── A0 action ────────────────────────────────────────────────────────────
    def _get_a0_action(self):
        """
        A0 acts as an attacker: chases ball from its own half and kicks
        toward goal_B0.  If a trained PPO policy is loaded, use that instead.
        """
        if self._a0_policy is not None:
            obs = self._get_obs_for_a0()
            a0_act, _ = self._a0_policy.predict(obs, deterministic=True)
            ddx, ddy  = DIR_MAP[int(a0_act[0])]
            return (ddx, ddy, int(a0_act[1]))

        return self._follower_action_for_a0()

    def _follower_action_for_a0(self):
        """Heuristic: chase ball, kick toward goal_B0, anti-own-goal guard."""
        ag = self.agents[1]
        b  = self.ball

        goal_B0_x    = self._goal_sign * self.HW
        own_goal_top = self.goal_center_y + self.goal_y
        own_goal_bot = self.goal_center_y - self.goal_y

        dist_to_ball = math.hypot(b.x - ag.x, b.y - ag.y)
        in_range     = dist_to_ball - ag.radius - b.radius < KICK_RANGE

        # Ray-test: would kicking cause own-goal (into goal_A0)?
        own_goal_A0_x = -self._goal_sign * self.HW
        would_own_goal = False
        if in_range:
            kx, ky = b.x - ag.x, b.y - ag.y
            kd = math.hypot(kx, ky)
            if kd > 0:
                nx, ny = kx / kd, ky / kd
                vx = b.xs + nx * KICK_STR
                vy = b.ys + ny * KICK_STR
                if abs(vx) > 0.01:
                    t = (own_goal_A0_x - b.x) / vx
                    if t > 0:
                        y_at = b.y + t * vy
                        if own_goal_bot < y_at < own_goal_top:
                            would_own_goal = True

        if would_own_goal:
            offset_y = self.goal_y + PLYR_R + 20
            target_x = b.x
            target_y = b.y + (offset_y if ag.y <= b.y else -offset_y)
            kick = 0
        else:
            target_x = b.x
            target_y = b.y
            kick = 1 if in_range else 0

        dx, dy = target_x - ag.x, target_y - ag.y
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            return (0, 0, kick)
        dx, dy = dx / dist, dy / dist

        best_dir_idx = 0; best_dot = -2.0
        for i, (mx, my) in enumerate(DIR_MAP):
            dot = dx * mx + dy * my
            if dot > best_dot:
                best_dot = dot; best_dir_idx = i

        out_dx, out_dy = DIR_MAP[best_dir_idx]
        return (out_dx, out_dy, kick)

    def _get_obs_for_a0(self) -> np.ndarray:
        """Build obs from A0's perspective (swap agents, flip team)."""
        orig_team  = self.team_id
        orig_flip  = self._flip
        self.team_id = 2 if self.team_id == 1 else 1
        self._flip   = -orig_flip
        self.agents[0], self.agents[1] = self.agents[1], self.agents[0]

        obs = self._get_obs()

        self.agents[0], self.agents[1] = self.agents[1], self.agents[0]
        self.team_id  = orig_team
        self._flip    = orig_flip
        return obs

    # ─── Observation ──────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """106-dim observation (same layout as env.py), B0-centric."""
        ball  = self.ball
        agent = self.agents[0]   # B0
        obs   = np.zeros(OBS_DIM, dtype=np.float32)

        bx, by   = ball.x,   ball.y
        bxs, bys = ball.xs,  ball.ys
        mx, my   = agent.x,  agent.y
        mxs, mys = agent.xs, agent.ys
        flip     = self._flip

        surf_dist = max(0.0, math.hypot(mx - bx, my - by) - PLYR_R - BALL_R)
        can_kick  = 1.0 if surf_dist < KICK_RANGE else 0.0

        i = 0

        # Section 1 — Field constants (4)
        obs[i] = self.goal_y / NORM;                    i += 1
        obs[i] = self.HH / NORM;                        i += 1
        obs[i] = self.HW / NORM;                        i += 1
        obs[i] = 0.0 if self.team_id == 1 else 1.0;    i += 1

        # Section 2 — Agent ↔ Ball (4)
        obs[i] = flip * (bx - mx) / NORM;   i += 1
        obs[i] = (by - my) / NORM;          i += 1
        obs[i] = surf_dist / DIAG;          i += 1
        obs[i] = can_kick;                  i += 1

        # Section 2b — Ball ↔ Goals (4)
        top_post_y = self.goal_center_y - self.goal_y
        bot_post_y = self.goal_center_y + self.goal_y
        nearest_post = top_post_y if abs(top_post_y - by) < abs(bot_post_y - by) else bot_post_y

        obs[i] = (self.HW - flip * bx) / NORM;      i += 1   # dx to opp goal line
        obs[i] = (nearest_post - by) / NORM;         i += 1   # dy to nearest post
        obs[i] = (-self.HW - flip * bx) / NORM;     i += 1   # dx to own goal line
        obs[i] = (nearest_post - by) / NORM;         i += 1   # dy to own post

        # Section 3 — Dynamic state (11)
        obs[i] = flip * bx / NORM;                          i += 1
        obs[i] = (by - self.goal_center_y) / NORM;          i += 1
        obs[i] = flip * bxs / MAX_SPEED;                    i += 1
        obs[i] = bys / MAX_SPEED;                           i += 1
        obs[i] = flip * mx / NORM;                          i += 1
        obs[i] = (my - self.goal_center_y) / NORM;          i += 1
        obs[i] = flip * mxs / MAX_SPEED;                    i += 1
        obs[i] = mys / MAX_SPEED;                           i += 1
        obs[i] = math.hypot(mxs, mys) / MAX_SPEED;         i += 1
        obs[i] = flip * (mxs - bxs) / MAX_SPEED;           i += 1
        obs[i] = (mys - bys) / MAX_SPEED;                   i += 1

        # Section 4 — Game state (2)
        obs[i] = max(0.0, 1.0 - self.step_count / self.max_steps); i += 1
        obs[i] = self.max_steps / MAX_STEPS_ALL_MODES;             i += 1

        assert i == 25, f"Obs pointer mismatch: {i}"
        # [25..105] remain 0 (no teammates/opponents in single-agent format)

        return obs

    def render(self):
        pass

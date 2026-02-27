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
        goal_y  : curriculum — Phase 0 enlarges goal toward [50%, 80%] × HH,
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
OBS_DIM   = 4 + 4 + 9 + 2 + N_TM * 9 + N_OPP * 9   # = 100

# ─────────────────────────────────────────────────────────────────────────────
# Training meta-constants
# ─────────────────────────────────────────────────────────────────────────────
FRAME_SKIP    = 6            # physics ticks per agent decision
PHYSICS_HZ    = 60

# Real futsal map presets: (HW, HH, goal_y, size_class)
MAP_PRESETS = [
    (368.0, 171.0, 64.0, '1v1'),   
    (520.0, 242.0, 76.0, '2v2'),   
    (401.0, 200.0, 70.0, '2v2'),   
]

# Field & Time Curriculum Thresholds
# Field & Time Curriculum Thresholds
CURRICULUM_PHASE2 =   300_000
CURRICULUM_PHASE3 = 1_000_000
CURRICULUM_TIME_1 = 2_000_000
CURRICULUM_TIME_2 = 3_000_000


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
class HaxballA0Env(gym.Env):
    """
    Stage A0: one RED agent, no opponents, score into RIGHT goal.

    Parameters
    ----------
    n_agents : int
        Number of agents in this team (affects field size range).
        1 → 1v1-sized field, 2 → 2v2-sized field.
    seed : int or None
        RNG seed.
    """

    metadata = {'render_modes': []}

    def __init__(self, n_agents: int = 1, seed: Optional[int] = None, legacy_obs: bool = False):
        super().__init__()

        self.n_agents = n_agents
        self.legacy_obs = legacy_obs
        self._rng = np.random.default_rng(seed)

        # Gymnasium spaces
        self.obs_dim = 100 if self.legacy_obs else OBS_DIM
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.obs_dim,), dtype=np.float32
        )
        # dim-0: movement (0-8), dim-1: kick (0/1)
        self.action_space = spaces.MultiDiscrete([9, 2])

        # Internal state (populated in reset)
        self.ball: Optional[Disc]    = None
        self.agents: list[Disc]      = []
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

        # Curriculum tracking: updated externally by GoalCurriculumCallback
        self.total_timesteps_elapsed: int = 0

        # Previous-step tracking for dense rewards
        self._prev_dist_to_ball = 0.0
        self._prev_ball_dist_to_goal = 0.0
        self._prev_ball_x       = 0.0
        self._prev_ball_speed   = 0.0

    # ─── reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        t = self.total_timesteps_elapsed
        
        # ── Time Curriculum ──
        if t < CURRICULUM_TIME_1:
            ep_secs = 15
        elif t < CURRICULUM_TIME_2:
            ep_secs = 10
        else:
            ep_secs = 7
        self.max_steps = ep_secs * PHYSICS_HZ // FRAME_SKIP

        # ── Field & Goal Curriculum ──
        # Always use full map size
        size_class = '1v1' if self.n_agents == 1 else '2v2'
        cands = [p for p in MAP_PRESETS if p[3] == size_class] or MAP_PRESETS
        preset = cands[int(self._rng.integers(0, len(cands)))]
        self.HW = float(preset[0])
        self.HH = float(preset[1])
        
        if t < CURRICULUM_PHASE2:
            # Goal is 70% to 90% of HH for the first 300k steps
            self.goal_y = float(self._rng.uniform(0.7, 0.9)) * self.HH
        else:
            self.goal_y = float(self._rng.uniform(30.0, 65.0))

        # Always Randomize Goal Center Y
        padding = 10.0 # Keep goal posts at least 10px inside the field
        max_center = max(0.0, self.HH - self.goal_y - padding)
        self.goal_center_y = float(self._rng.uniform(-max_center, max_center))

        self.step_count = 0

        # Randomize team each episode → shared policy via x-flip
        self.team_id      = int(self._rng.integers(1, 3))   # 1 or 2
        self._flip        = 1.0 if self.team_id == 1 else -1.0
        self._attack_sign = 1   if self.team_id == 1 else -1

        # Spawn ball (random inside 70% of field)
        bx = float(self._rng.uniform(-self.HW * 0.7, self.HW * 0.7))
        by = float(self._rng.uniform(-self.HH * 0.7, self.HH * 0.7))
        self.ball = Disc(bx, by, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)

        # Spawn agents (no overlap with ball or each other)
        self.agents = []
        for _ in range(self.n_agents):
            pos = self._safe_spawn()
            self.agents.append(
                Disc(pos[0], pos[1], 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)
            )

        # Init dense-reward tracking
        a = self.agents[0]
        self._prev_dist_to_ball = math.hypot(a.x - self.ball.x, a.y - self.ball.y)
        
        # Distance from ball to ATTACK goal center (x = HW * attack_sign, y = goal_center_y)
        goal_x = self.HW * self._attack_sign
        self._prev_ball_dist_to_goal = math.hypot(goal_x - self.ball.x, self.goal_center_y - self.ball.y)
        
        self._prev_ball_x       = self.ball.x
        self._prev_ball_speed   = math.hypot(self.ball.xs, self.ball.ys)

        return self._get_obs(), {}

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

        # Run FRAME_SKIP ticks
        goal_result = 0   # 0=none, 1=left(own), 2=right(scored)
        for _ in range(FRAME_SKIP):
            result = self._tick(dx, dy, kick)
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
        atk = self._attack_sign

        if goal_result == 2:
            # Scored into right goal ✅
            reward = 5.0  # Sparse reward for scoring
            terminated = True

        else:
            dist_to_ball = math.hypot(a.x - self.ball.x, a.y - self.ball.y)
            self._prev_dist_to_ball = dist_to_ball
            
            goal_x = self.HW * atk
            ball_dist_to_goal = math.hypot(goal_x - self.ball.x, self.goal_center_y - self.ball.y)
            self._prev_ball_dist_to_goal = ball_dist_to_goal

            ball_speed = math.hypot(self.ball.xs, self.ball.ys)
            self._prev_ball_speed = ball_speed
            
            # Time penalty if step > 60
            if self.step_count > 60:
                # Penalty: step 61 = -0.001, step 62 = -0.0012, step 63 = -0.0014...
                # Formula: 0.001 + (step_count - 61) * 0.0002
                steps_over = self.step_count - 60
                reward -= (0.001 + (steps_over - 1) * 0.0002)

        # Timeout
        truncated = False
        if not terminated and self.step_count >= self.max_steps:
            # No negative reward as requested
            reward = 0.0
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {}

    # ─── Physics ──────────────────────────────────────────────────────────────
    def _tick(self, dx: float, dy: float, kick: int) -> int:
        """
        One physics tick.
        Returns 0=normal | 1=left-goal (own) | 2=right-goal (scored).
        Exact port from test_index.html physicsStep().
        """
        ball   = self.ball
        agents = self.agents
        HW, HH = self.HW, self.HH

        # 1. Kick (before movement)
        if kick:
            for ag in agents:
                dx_b = ball.x - ag.x
                dy_b = ball.y - ag.y
                dist = math.hypot(dx_b, dy_b)
                if dist > 0 and dist - ag.radius - ball.radius < KICK_RANGE:
                    nx, ny = dx_b / dist, dy_b / dist
                    ball.xs += nx * KICK_STR
                    ball.ys += ny * KICK_STR

        # 2. Acceleration
        ln = math.hypot(dx, dy)
        acc = PLYR_KICK_ACC if kick else PLYR_ACC
        if ln > 0:
            ndx, ndy = dx / ln, dy / ln
            for ag in agents:
                ag.xs += ndx * acc
                ag.ys += ndy * acc

        # 3. Move all
        ball.x += ball.xs; ball.y += ball.ys
        for ag in agents:
            ag.x += ag.xs; ag.y += ag.ys

        # 4. Disc-Disc collisions (Players and Ball)
        all_discs = [ball] + agents
        n = len(all_discs)
        for i in range(n):
            for j in range(i + 1, n):
                _resolve_dd(all_discs[i], all_discs[j])

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
        obs[i] = 0.0;                         i += 1  # possession (no opps)

        # Sections 5 & 6 — Teammates × 4 + Opponents × 5, all zeros in A0
        # (i advances to 100 through the zero-filled array)

        assert i == 21, f"Obs pointer mismatch before padding: {i}"
        # Remaining [19..99] stay 0 (no teammates / opponents in A0)

        return obs

    def render(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Standalone physics helper (module-level for speed)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_dd(da: Disc, db: Disc) -> None:
    """Exact port of resolveDDCollision from test_index.html."""
    ddx = da.x - db.x; ddy = da.y - db.y
    dist = math.hypot(ddx, ddy)
    r_sum = da.radius + db.radius
    if 0 < dist <= r_sum:
        nx, ny  = ddx / dist, ddy / dist
        
        # Handle immovable objects (Pole has imass == 0)
        imass_sum = da.imass + db.imass
        if imass_sum == 0:
            return  # both immovable, do nothing
            
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

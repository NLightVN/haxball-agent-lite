import re

with open("training/env.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Clean __init__
init_old = r"""    def __init__\(self, phase: str = 'A0', n_agents: int = 1, seed: Optional\[int\] = None, legacy_obs: bool = False\):
        super\(\)\.__init__\(\)

        self\.phase = phase
        self\.n_agents = n_agents
        self\.legacy_obs = legacy_obs
        self\._rng = np\.random\.default_rng\(seed\)
        
        self\.frame_skip = 3 if self\.phase in \('A1\.2', 'A0\.1', 'A1'\) else FRAME_SKIP

        # Gymnasium spaces
        self\.obs_dim = 100 if self\.legacy_obs else OBS_DIM"""

init_new = """    def __init__(self, n_agents: int = 1, seed: Optional[int] = None):
        super().__init__()

        self.n_agents = n_agents
        self._rng = np.random.default_rng(seed)
        
        # MARL physics ticks per step
        self.frame_skip = 3

        # Gymnasium spaces
        self.obs_dim = OBS_DIM"""

content = re.sub(init_old, init_new, content)

# 2. Clean reset() entirely.
# Replace from `def reset` to `def _reset_positions`
reset_match = re.search(r'    def reset\(self.*?\n    def _reset_positions\(self\):', content, re.DOTALL)
if reset_match:
    new_reset = """    def reset(self, *, seed=None, options=None):
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
        self._prev_ball_speed   = math.hypot(self.ball.xs, self.ball.ys)
        
        self.last_touch = None
        self.last_touch_team = None
        self.possession_history.clear()
        self.investment_sequence.clear()
        self.opp_possession_time = 0.0
        self.dribble_start_time = 0.0
        self.self_pass_shooter = None
        self.self_pass_active = False

        return self._get_obs(), {}

    def _reset_positions(self):"""
    content = content.replace(reset_match.group(0), new_reset)

# 3. Clean _reset_positions()
# Replace the body to just spawn 1v1 and pick a random opponent from the 3 random algorithms.
reset_pos_match = re.search(r'    def _reset_positions\(self\):.*?def _safe_spawn', content, re.DOTALL)
if reset_pos_match:
    new_reset_pos = """    def _reset_positions(self):
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

    def _safe_spawn"""
    content = content.replace(reset_pos_match.group(0), new_reset_pos)

with open("training/env.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Cleaned env.py")

import re
import os

with open("training/env.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update OBS_DIM
content = re.sub(
    r'OBS_DIM\s*=\s*4 \+ 4 \+ 4 \+ 11 \+ 2 \+ N_TM \* 9 \+ N_OPP \* 9\s*# = 106',
    'OBS_DIM = 4 + 4 + 4 + 11 + 2 + N_TM * 9 + N_OPP * 9 + 3 + 3 + 1 + 4 + N_TM * 4 + N_OPP * 4  # = 153',
    content
)

# 2. Add deque import
content = content.replace("import numpy as np", "import numpy as np\nfrom collections import deque")

# 3. Add sequence vars to __init__
init_old = "self.scores = [0, 0]          # RED, BLUE\n        self.last_touch = None"
init_new = """self.scores = [0, 0]          # RED, BLUE
        self.last_touch = None
        self.last_touch_team = None
        
        # MARL Reward Logic
        self.possession_history = deque(maxlen=15)  # 0.25s at 60Hz physics (FRAME_SKIP is used, but physics ticks are 60Hz. step() represents frame_skip ticks. Wait, history should be in terms of real time. If step is 6 ticks, 15 ticks = 2.5 steps. We'll store (time_sec, team_id) and clean old ones.)
        self.investment_sequence = []
        self.opp_possession_time = 0.0
        self.dribble_start_time = 0.0
        self.self_pass_shooter = None
        self.self_pass_active = False"""
content = content.replace(init_old, init_new)

# 4. Modify _tick to track exact player who touched
tick_old = """        if touched_agent and touched_opp:
            # Both touched in same tick, ambiguous but let's favor the last one or None
            # Standard is just to update who hit it. Let's say agent wins ties.
            self.last_touch_this_tick = 'A'
        elif touched_agent:
            self.last_touch_this_tick = 'A'
        elif touched_opp:
            self.last_touch_this_tick = 'O'
        else:
            self.last_touch_this_tick = None"""

tick_new = """        # Record exact player touch in _tick
        self.last_touch_player_this_tick = None
        self.last_touch_team_this_tick = None
        for i in range(n):
            for j in range(i + 1, n):
                if _resolve_dd(all_discs[i], all_discs[j]):
                    if (i == 0 and j > 0) or (j == 0 and i > 0):
                        idx = max(i, j) - 1 # idx in self.agents
                        self.last_touch_player_this_tick = idx
                        # Determine team of idx
                        # agents[0] is team_id. agents[1] is opp_id.
                        # Since it's currently 1v1, idx 0 is team_id, idx 1 is opp team.
                        self.last_touch_team_this_tick = self.team_id if idx == 0 else (2 if self.team_id == 1 else 1)
"""
# Wait! In _tick, the loops for _resolve_dd are ALREADY THERE!
# I should replace the ENTIRE block 4 of _tick.
content = re.sub(
    r'        # 4. Disc-Disc collisions.*?        # 4b\. Goal Poles Collisions',
    r'''        # 4. Disc-Disc collisions (Players and Ball)
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
                            
        # 4b. Goal Poles Collisions''',
    content,
    flags=re.DOTALL
)

# 5. Overwrite the step method completely
# We'll use a regex to replace from `def step(self, action):` to `return self._get_obs(), float(reward), terminated, truncated, info`
step_match = re.search(r'    def step\(self, action\):.*?return self\._get_obs\(\), float\(reward\), terminated, truncated, info', content, re.DOTALL)
if step_match:
    old_step = step_match.group(0)
    
    with open("marl_step.py", "r", encoding="utf-8") as fm:
        new_step = fm.read()
        
    content = content.replace(old_step, new_step)

# 6. Overwrite _get_obs
obs_match = re.search(r'    def _get_obs\(self\) -> np\.ndarray:.*?return obs', content, re.DOTALL)
if obs_match:
    with open("marl_obs.py", "r", encoding="utf-8") as fo:
        new_obs = fo.read()
    content = content.replace(obs_match.group(0), new_obs)

with open("training/env.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Rewrote env.py")

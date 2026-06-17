import re

with open('single-agent-files/training/env.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update OBS_DIM
content = re.sub(
    r'OBS_DIM\s*=\s*4 \+ 4 \+ 4 \+ 11 \+ 2 \+ N_TM \* 9 \+ N_OPP \* 9\s*# = 106',
    'OBS_DIM = 4 + 4 + 4 + 11 + 2 + N_TM * 9 + N_OPP * 9 + 3 + 4 + N_TM * 4 + N_OPP * 4  # = 149',
    content
)

# 2. Add imports
content = re.sub(
    r'import numpy as np',
    'import numpy as np\nfrom collections import deque',
    content
)

# 3. Add sequence vars to __init__
init_hook = r'self.scores = \[0, 0\]          # RED, BLUE\n        self.last_touch = None'
new_init = '''self.scores = [0, 0]          # RED, BLUE
        self.last_touch = None
        
        # MARL Reward logic vars
        self.possession_history = deque(maxlen=15) # 0.25s at 60Hz
        self.investment_sequence = []
        self.opp_possession_timer = 0.0
        self.self_pass_shooter = None
        self.self_pass_target = None
        self.dribble_timer = 0.0
'''
content = content.replace(init_hook, new_init)

# 4. Reset sequence vars in reset
reset_hook = r'self.last_touch = None\n        self._a0_1_best_ball_dist_to_goal = self._prev_ball_dist_to_goal'
new_reset = '''self.last_touch = None
        self.possession_history.clear()
        self.investment_sequence.clear()
        self.opp_possession_timer = 0.0
        self.self_pass_shooter = None
        self.self_pass_target = None
        self.dribble_timer = 0.0
        self._a0_1_best_ball_dist_to_goal = self._prev_ball_dist_to_goal'''
content = content.replace(reset_hook, new_reset)

# 5. Overwrite _tick to return who touched the ball instead of just A or O
tick_touch_hook = r'''        if touched_agent and touched_opp:
            # Both touched in same tick, ambiguous but let's favor the last one or None
            # Standard is just to update who hit it. Let's say agent wins ties.
            self.last_touch_this_tick = 'A'
        elif touched_agent:
            self.last_touch_this_tick = 'A'
        elif touched_opp:
            self.last_touch_this_tick = 'O'
        else:
            self.last_touch_this_tick = None'''

new_tick_touch = '''        # Record the exact player ID (index in self.agents) who touched
        self.last_touch_this_tick = None
        for i in range(n):
            for j in range(i + 1, n):
                if _resolve_dd(all_discs[i], all_discs[j]):
                    if i == 0 and j > 0:
                        self.last_touch_this_tick = j - 1
                    elif j == 0 and i > 0:
                        self.last_touch_this_tick = i - 1'''
# Actually wait! The `_resolve_dd` was already processed inside the loop above! 
# Let me replace the entire _tick collision loop

# Wait, this is getting complex to do via python string replace.

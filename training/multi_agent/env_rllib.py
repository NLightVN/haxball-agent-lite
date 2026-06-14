import math
import numpy as np
from typing import Optional, Dict, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces

# Import constants and physics from existing env_multi
from training.multi_agent.env_multi import (
    Disc, _resolve_dd, _dist_to_goal_segment,
    PHYSICS_HZ, FRAME_SKIP, MAP_PRESETS,
    BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP,
    PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP,
    PLYR_ACC, PLYR_KICK_ACC, MAX_SPEED, KICK_RANGE, KICK_STR,
    POLE_R, POLE_IMASS, POLE_BCOEF, OUTER_PAD, GOAL_DEPTH,
    DIR_MAP, OBS_DIM
)

NORM = 400.0
DIAG = 800.0
N_TM = 3
N_OPP = 4
MAX_STEPS_ALL_MODES = 2 * 60 * (PHYSICS_HZ // FRAME_SKIP) # 2 mins

class HaxballRLlibEnv(MultiAgentEnv):
    """
    Native Ray RLlib Multi-Agent Environment for 2v2.
    Agents: "red_0", "red_1", "blue_0", "blue_1".
    RED attacks RIGHT (+HW). BLUE attacks LEFT (-HW).
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        self.n_players = 2 # 2v2
        
        self.agent_ids = [f"red_{i}" for i in range(self.n_players)] + [f"blue_{i}" for i in range(self.n_players)]
        self._agent_ids = set(self.agent_ids)
        
        # RLlib requires these
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([9, 2])
        self._rng = np.random.default_rng()
        
        # State
        self.ball: Optional[Disc] = None
        self.agents: Dict[str, Disc] = {}
        self.HW = 0.0
        self.HH = 0.0
        self.goal_y = 64.0
        self.goal_center_y = 0.0
        self.step_count = 0
        self.max_steps = 150 * (PHYSICS_HZ // FRAME_SKIP) # 2.5 minutes
        
        self.scores = [0, 0] # RED, BLUE
        
        # Dense reward tracking (per agent)
        self._prev_dist_to_goal = {} # ball to opponent goal from agent's perspective
        
        # Possession & Investment
        self.tick_count = 0
        self.possession_team = 0
        self.tentative_team = 0
        self.tentative_start_tick = 0
        self.tentative_last_tick = 0
        
        self._inv_pool = {} # {agent_id: credit}
        self._inv_events = {ag_id: [] for ag_id in self.agent_ids} # {agent_id: [(step, credit)]}
        self._inv_last_toucher = None
        self._inv_step = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
        preset = MAP_PRESETS[1] # standard 2v2/3v3 preset
        self.HW = float(preset[0])
        self.HH = float(preset[1])
        self.goal_y = float(preset[2])
        self.goal_center_y = 0.0
        
        self.step_count = 0
        self.scores = [0, 0]
        
        # Spawn ball
        self.ball = Disc(0.0, 0.0, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
        
        # Spawn agents
        self.agents = {}
        for ag_id in self.agent_ids:
            team = 1 if "red" in ag_id else 2
            # RED spawned on left (-HW), BLUE on right (+HW)
            if team == 1:
                x = float(self._rng.uniform(-self.HW + PLYR_R, -PLYR_R))
            else:
                x = float(self._rng.uniform(PLYR_R, self.HW - PLYR_R))
            y = float(self._rng.uniform(-self.HH + PLYR_R, self.HH - PLYR_R))
            self.agents[ag_id] = Disc(x, y, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP, team=team)
            
            atk = 1 if team == 1 else -1
            self._prev_dist_to_goal[ag_id] = _dist_to_goal_segment(self.ball.x, self.ball.y, self.HW * atk, self.goal_y, -self.goal_y)

        # Reset possession
        self.tick_count = 0
        self.possession_team = 0
        self.tentative_team = 0
        self.tentative_start_tick = 0
        self.tentative_last_tick = 0
        
        self._inv_pool = {}
        self._inv_events = {ag_id: [] for ag_id in self.agent_ids}
        self._inv_last_toucher = None
        self._inv_step = 0
        
        obs_dict = {ag_id: self._get_obs(ag_id) for ag_id in self.agent_ids}
        return obs_dict, {}

    def step(self, action_dict):
        # 1. Parse actions
        agent_actions = []
        ag_id_list = list(self.agents.keys())
        
        for ag_id in ag_id_list:
            if ag_id in action_dict:
                act = action_dict[ag_id]
                dir_idx = int(act[0])
                kick = int(act[1])
            else:
                dir_idx = 0
                kick = 0
                
            dx, dy = DIR_MAP[dir_idx]
            flip = 1.0 if self.agents[ag_id].team == 1 else -1.0
            agent_actions.append((dx * flip, dy, kick))
            
        # 2. Physics Ticks
        goal_result = 0
        for _ in range(FRAME_SKIP):
            result = self._tick(ag_id_list, agent_actions)
            if result != 0:
                goal_result = result
                break
                
        self.step_count += 1
        self._inv_step += 1
        
        # 3. Calculate Rewards & State
        rewards = {ag_id: 0.0 for ag_id in self.agent_ids}
        infos = {ag_id: {} for ag_id in self.agent_ids}
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        
        goal_scored = False
        if goal_result == 2: # Scored into right goal (RED scores)
            self.scores[0] += 1
            goal_scored = True
            self._handle_goal_rewards(rewards, infos, scoring_team=1)
        elif goal_result == 1: # Scored into left goal (BLUE scores)
            self.scores[1] += 1
            goal_scored = True
            self._handle_goal_rewards(rewards, infos, scoring_team=2)
            
        if goal_scored:
            self._reset_positions_after_goal()
            
        # Dense rewards
        for ag_id, ag in self.agents.items():
            atk = 1 if ag.team == 1 else -1
            cur_dist = _dist_to_goal_segment(self.ball.x, self.ball.y, self.HW * atk, self.goal_y, -self.goal_y)
            delta = self._prev_dist_to_goal[ag_id] - cur_dist
            if not goal_scored:
                rewards[ag_id] += delta * (1.0 / (2.0 * self.HW))
            self._prev_dist_to_goal[ag_id] = cur_dist

        if self.scores[0] >= 1 or self.scores[1] >= 1:
            terminateds = {"__all__": True}
            
        if not terminateds["__all__"] and self.step_count >= self.max_steps:
            truncateds = {"__all__": True}
            
        obs_dict = {ag_id: self._get_obs(ag_id) for ag_id in self.agent_ids}
        
        return obs_dict, rewards, terminateds, truncateds, infos

    def _handle_goal_rewards(self, rewards, infos, scoring_team):
        # Base rewards
        for ag_id, ag in self.agents.items():
            if ag.team == scoring_team:
                # Investment Algorithm Retroactive Payoff
                credit = self._inv_pool.get(ag_id, 0.0)
                was_scorer = (self._inv_last_toucher == ag_id)
                scorer_base = 30.0 * 0.5 if was_scorer else 0.0
                rewards[ag_id] += scorer_base
                
                retro_total = (credit - (0.5 if was_scorer else 0.0)) * 30.0
                if retro_total > 1e-6 and len(self._inv_events[ag_id]) > 0:
                    weights = [cr for (_, cr) in self._inv_events[ag_id]]
                    total_w = sum(weights)
                    invest_credits = [
                        (self._inv_step - ev_step, (w / total_w) * retro_total)
                        for (ev_step, w) in self._inv_events[ag_id]
                    ]
                    infos[ag_id]["investment_credit"] = invest_credits
                elif not was_scorer and credit == 0.0:
                    rewards[ag_id] += 0.0 # No participation
            else:
                rewards[ag_id] -= 10.0 # Penalty for conceding
                
    def _reset_positions_after_goal(self):
        self.ball = Disc(0.0, 0.0, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
        for ag_id, ag in self.agents.items():
            if ag.team == 1:
                ag.x = float(self._rng.uniform(-self.HW + PLYR_R, -PLYR_R))
            else:
                ag.x = float(self._rng.uniform(PLYR_R, self.HW - PLYR_R))
            ag.y = float(self._rng.uniform(-self.HH + PLYR_R, self.HH - PLYR_R))
            ag.xs = ag.ys = 0.0
            
            atk = 1 if ag.team == 1 else -1
            self._prev_dist_to_goal[ag_id] = _dist_to_goal_segment(self.ball.x, self.ball.y, self.HW * atk, self.goal_y, -self.goal_y)

    def _tick(self, ag_id_list, agent_actions) -> int:
        ball = self.ball
        HW, HH = self.HW, self.HH
        self.tick_count += 1
        interacting_teams = set()
        
        action_order = list(range(len(agent_actions)))
        self._rng.shuffle(action_order)

        # 1. Kick
        for i in action_order:
            ag = self.agents[ag_id_list[i]]
            dx, dy, kick = agent_actions[i]
            if kick:
                dx_b = ball.x - ag.x
                dy_b = ball.y - ag.y
                dist = math.hypot(dx_b, dy_b)
                if dist > 0 and dist - ag.radius - ball.radius < KICK_RANGE:
                    nx, ny = dx_b / dist, dy_b / dist
                    ball.xs += nx * KICK_STR
                    ball.ys += ny * KICK_STR
                    interacting_teams.add(ag.team)

        # 2. Accel
        for i in action_order:
            ag = self.agents[ag_id_list[i]]
            dx, dy, kick = agent_actions[i]
            ln = math.hypot(dx, dy)
            acc = PLYR_KICK_ACC if kick else PLYR_ACC
            if ln > 0:
                ag.xs += (dx / ln) * acc
                ag.ys += (dy / ln) * acc

        # 3. Move
        ball.x += ball.xs; ball.y += ball.ys
        for ag in self.agents.values():
            ag.x += ag.xs; ag.y += ag.ys

        # 4. Collisions
        all_discs = [ball] + list(self.agents.values())
        touching_ag_ids = set()
        
        for i in range(len(all_discs)):
            for j in range(i + 1, len(all_discs)):
                if _resolve_dd(all_discs[i], all_discs[j]):
                    if i == 0 or j == 0:
                        ag_disc = all_discs[max(i, j)]
                        # find ag_id
                        for ag_id, a in self.agents.items():
                            if a is ag_disc:
                                touching_ag_ids.add(ag_id)
                                interacting_teams.add(a.team)
                                break

        # Kick counts as touch for possession
        for i in action_order:
            ag = self.agents[ag_id_list[i]]
            _, _, kick = agent_actions[i]
            if kick:
                dist_k = math.hypot(ball.x - ag.x, ball.y - ag.y)
                if dist_k > 0 and dist_k - ag.radius - ball.radius < KICK_RANGE:
                    touching_ag_ids.add(ag_id_list[i])

        # Possession logic
        prev_pos = self.possession_team
        if interacting_teams:
            if self.possession_team == 0:
                self.possession_team = list(interacting_teams)[0]
            elif self.possession_team in interacting_teams:
                self.tentative_team = 0
            else:
                other = list(interacting_teams)[0]
                if self.tentative_team != other:
                    self.tentative_team = other
                    self.tentative_start_tick = self.tick_count
                self.tentative_last_tick = self.tick_count
                
                if self.tentative_last_tick - self.tentative_start_tick >= 30:
                    self.possession_team = self.tentative_team
                    self.tentative_team = 0

        # Investment Update
        if touching_ag_ids:
            # We assume investment works per team. We handle the team that currently holds possession.
            # If possession changed, reset pool
            if self.possession_team != prev_pos and prev_pos != 0:
                self._inv_pool = {}
                self._inv_events = {a: [] for a in self.agent_ids}
                self._inv_last_toucher = None
            
            # Update for the possessing team
            possessing_touchers = [aid for aid in touching_ag_ids if self.agents[aid].team == self.possession_team]
            for ag_id in possessing_touchers:
                if self._inv_last_toucher is None:
                    self._inv_pool = {ag_id: 1.0}
                    self._inv_last_toucher = ag_id
                elif self._inv_last_toucher != ag_id:
                    # Pass occurred
                    prev_toucher = self._inv_last_toucher
                    for k in list(self._inv_pool.keys()):
                        self._inv_pool[k] *= 0.5
                    self._inv_pool[ag_id] = self._inv_pool.get(ag_id, 0.0) + 0.5
                    
                    # Log event for passer
                    cr = self._inv_pool.get(prev_toucher, 0.0)
                    self._inv_events[prev_toucher].append((self._inv_step, cr))
                    self._inv_last_toucher = ag_id

        # Wall/Goal Collisions
        poles = [
            Disc( HW, self.goal_center_y + self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
            Disc( HW, self.goal_center_y - self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
            Disc(-HW, self.goal_center_y + self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
            Disc(-HW, self.goal_center_y - self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
        ]
        for p in poles:
            for ag in self.agents.values(): _resolve_dd(ag, p)
            _resolve_dd(ball, p)

        if ball.y - ball.radius < -HH: ball.y = -HH + ball.radius; ball.ys *= -ball.bcoef
        if ball.y + ball.radius > HH: ball.y = HH - ball.radius; ball.ys *= -ball.bcoef

        if ball.x - ball.radius < -HW:
            if self.goal_center_y - self.goal_y < ball.y < self.goal_center_y + self.goal_y: return 1
            ball.x = -HW + ball.radius; ball.xs *= -ball.bcoef
        elif ball.x + ball.radius > HW:
            if self.goal_center_y - self.goal_y < ball.y < self.goal_center_y + self.goal_y: return 2
            ball.x = HW - ball.radius; ball.xs *= -ball.bcoef

        max_y = HH + OUTER_PAD; max_x = HW + GOAL_DEPTH
        for ag in self.agents.values():
            if ag.x - ag.radius < -max_x: ag.x = -max_x + ag.radius; ag.xs = max(0, ag.xs)
            elif ag.x + ag.radius > max_x: ag.x = max_x - ag.radius; ag.xs = min(0, ag.xs)
            if ag.y - ag.radius < -max_y: ag.y = -max_y + ag.radius; ag.ys = max(0, ag.ys)
            elif ag.y + ag.radius > max_y: ag.y = max_y - ag.radius; ag.ys = min(0, ag.ys)

        ball.xs *= ball.damp; ball.ys *= ball.damp
        for ag in self.agents.values(): ag.xs *= ag.damp; ag.ys *= ag.damp

        return 0

    def _get_obs(self, ag_id: str) -> np.ndarray:
        ball = self.ball
        agent = self.agents[ag_id]
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        flip = 1.0 if agent.team == 1 else -1.0
        
        bx, by = ball.x, ball.y
        bxs, bys = ball.xs, ball.ys
        mx, my = agent.x, agent.y
        mxs, mys = agent.xs, agent.ys
        
        surf_dist = max(0.0, math.hypot(mx - bx, my - by) - PLYR_R - BALL_R)
        can_kick = 1.0 if surf_dist < KICK_RANGE else 0.0
        
        i = 0
        obs[i] = self.goal_y / NORM; i+=1
        obs[i] = self.HH / NORM; i+=1
        obs[i] = self.HW / NORM; i+=1
        
        my_dist = math.hypot(mx - bx, my - by)
        team_dists = [math.hypot(a.x - bx, a.y - by) for aid, a in self.agents.items() if a.team == agent.team]
        all_dists = [math.hypot(a.x - bx, a.y - by) for a in self.agents.values()]
        
        obs[i] = sum(1 for d in team_dists if d < my_dist); i+=1
        obs[i] = 1.0 if sum(1 for d in all_dists if d < my_dist) == 0 else 0.0; i+=1
        
        obs[i] = flip * (bx - mx) / NORM; i+=1
        obs[i] = (by - my) / NORM; i+=1
        obs[i] = surf_dist / DIAG; i+=1
        obs[i] = can_kick; i+=1
        
        obs[i] = (self.HW - flip * bx) / NORM; i+=1
        obs[i] = (self.goal_center_y - by) / NORM; i+=1
        obs[i] = (-self.HW - flip * bx) / NORM; i+=1
        obs[i] = (self.goal_center_y - by) / NORM; i+=1
        
        obs[i] = flip * bx / NORM; i+=1
        obs[i] = (by - self.goal_center_y) / NORM; i+=1
        obs[i] = flip * bxs / MAX_SPEED; i+=1
        obs[i] = bys / MAX_SPEED; i+=1
        obs[i] = flip * mx / NORM; i+=1
        obs[i] = (my - self.goal_center_y) / NORM; i+=1
        obs[i] = flip * mxs / MAX_SPEED; i+=1
        obs[i] = mys / MAX_SPEED; i+=1
        obs[i] = math.hypot(mxs, mys) / MAX_SPEED; i+=1
        obs[i] = flip * (mxs - bxs) / MAX_SPEED; i+=1
        obs[i] = (mys - bys) / MAX_SPEED; i+=1
        
        obs[i] = max(0.0, 1.0 - self.step_count / self.max_steps); i+=1
        obs[i] = 1.0; i+=1
        
        # Others
        tm_idx, opp_idx = 26, 62
        tm_count, opp_count = 0, 0
        for aid, ag in self.agents.items():
            if aid == ag_id: continue
            if ag.team == agent.team:
                if tm_count >= N_TM: continue
                idx = tm_idx + tm_count * 9
                tm_count += 1
            else:
                if opp_count >= N_OPP: continue
                idx = opp_idx + opp_count * 9
                opp_count += 1
                
            obs[idx] = flip * ag.x / NORM; idx+=1
            obs[idx] = ag.y / NORM; idx+=1
            obs[idx] = flip * ag.xs / MAX_SPEED; idx+=1
            obs[idx] = ag.ys / MAX_SPEED; idx+=1
            obs[idx] = flip * (ag.x - mx) / NORM; idx+=1
            obs[idx] = (ag.y - my) / NORM; idx+=1
            obs[idx] = flip * (bx - ag.x) / NORM; idx+=1
            obs[idx] = (by - ag.y) / NORM; idx+=1
            obs[idx] = max(0.0, math.hypot(ag.x - bx, ag.y - by) - PLYR_R - BALL_R) / DIAG; idx+=1
            
        return obs

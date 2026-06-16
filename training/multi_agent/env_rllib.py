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
        self._prev_dist_to_goal = {} # kept for potential future use
        
        # Possession & Investment
        self.tick_count = 0
        self.possession_team = 0
        self.tentative_team = 0
        self.tentative_start_tick = 0
        self.tentative_last_tick = 0
        
        # Investment: ordered chain [(ag_id, ratio, step_when_entered)]
        # Latest entry always has ratio 1.0; each prior entry is halved on every new touch.
        self._inv_chain: list = []
        self._inv_last_toucher = None
        self._inv_step = 0

        self.global_timesteps = 0
        self.match_mode = "2v2"
        self.active_agent_ids = self.agent_ids

    def set_timesteps(self, ts: int):
        self.global_timesteps = ts
        # Possession loss event: (losing_team, last_toucher_id) or None
        self._possession_loss_event = None

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
        
        # Decide match mode based on curriculum
        # < 10M: 50% 1v1, 50% 2v2
        # >= 10M: 20% 1v1, 80% 2v2
        prob_1v1 = 0.5 if self.global_timesteps < 10_000_000 else 0.2
        self.match_mode = "1v1" if self._rng.random() < prob_1v1 else "2v2"

        if self.match_mode == "1v1":
            self.active_agent_ids = ["red_0", "blue_0"]
        else:
            self.active_agent_ids = self.agent_ids

        # Spawn ball
        self.ball = Disc(0.0, 0.0, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
        
        # Spawn agents
        self.agents = {}
        for ag_id in self.active_agent_ids:
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
        
        self._inv_chain = []
        self._inv_last_toucher = None
        self._inv_step = 0
        
        obs_dict = {ag_id: self._get_obs(ag_id) for ag_id in self.active_agent_ids}
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
        self._possession_loss_event = None  # reset each step
        for _ in range(FRAME_SKIP):
            result = self._tick(ag_id_list, agent_actions)
            if result != 0:
                goal_result = result
                break
                
        self.step_count += 1
        self._inv_step += 1
        
        # 3. Calculate Rewards & State
        rewards = {ag_id: 0.0 for ag_id in self.active_agent_ids}
        infos = {ag_id: {} for ag_id in self.active_agent_ids}
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
            self._prev_dist_to_goal[ag_id] = cur_dist

        # Positional Possession Reward (per agent)
        # Carrier gets full reward; other teammates get 10%
        if not goal_scored:
            pos_reward_by_agent = self._calc_positional_possession_reward()
            for ag_id in self.active_agent_ids:
                if ag_id in pos_reward_by_agent:
                    rewards[ag_id] += pos_reward_by_agent[ag_id]

        if self.scores[0] >= 1 or self.scores[1] >= 1:
            terminateds = {"__all__": True}
            
        if not terminateds["__all__"] and self.step_count >= self.max_steps:
            truncateds = {"__all__": True}
            
        obs_dict = {ag_id: self._get_obs(ag_id) for ag_id in self.active_agent_ids}
        
        return obs_dict, rewards, terminateds, truncateds, infos

    def _calc_positional_possession_reward(self) -> dict:
        """
        Returns {ag_id: reward} for the positional possession reward.

        Logic (per possessing team):
          - Find the carrier: team member nearest to the ball.
          - Count how many opponents the carrier has bypassed along the x-axis
            by at least BYPASS_GAP units in the attack direction.
          - 0 bypassed → +0.0001   (carrier); +0.00001 (teammates)
          - 1 bypassed → +0.0005   (carrier); +0.00005 (teammates)
          - 2 bypassed → +0.005    (carrier); +0.0005  (teammates)

        Carrier gets full reward; other teammates get 10% of that amount.
        Only awarded when self.possession_team != 0.
        """
        BYPASS_GAP = 15.0
        # Carrier rewards
        REWARD_0 = 0.0001
        REWARD_1 = 0.0005
        REWARD_2 = 0.005
        
        # Teammate rewards
        REWARD_0_TM = 0.0
        REWARD_1_TM = 0.00001
        REWARD_2_TM = 0.00005

        result = {}
        if self.possession_team == 0:
            return result

        pteam = self.possession_team
        atk = 1 if pteam == 1 else -1   # +1 means RED attacks right (+x)

        # Find carrier: team member nearest to the ball
        ball_x, ball_y = self.ball.x, self.ball.y
        carrier_id = None
        carrier = None
        carrier_dist = float('inf')
        for ag_id, ag in self.agents.items():
            if ag.team != pteam:
                continue
            d = math.hypot(ag.x - ball_x, ag.y - ball_y)
            if d < carrier_dist:
                carrier_dist = d
                carrier = ag
                carrier_id = ag_id

        if carrier is None:
            return result

        # Sanity guard: only reward when carrier is actually close
        if carrier_dist > self.HW:
            return result

        # Count bypassed opponents
        opp_team = 2 if pteam == 1 else 1
        bypassed = 0
        for ag in self.agents.values():
            if ag.team != opp_team:
                continue
            gap = atk * (carrier.x - ag.x)   # positive = carrier is ahead of this opponent
            if gap >= BYPASS_GAP:
                bypassed += 1

        if bypassed >= 2:
            carrier_reward = REWARD_2
            teammate_reward = REWARD_2_TM
        elif bypassed == 1:
            carrier_reward = REWARD_1
            teammate_reward = REWARD_1_TM
        else:
            carrier_reward = REWARD_0
            teammate_reward = REWARD_0_TM

        for ag_id, ag in self.agents.items():
            if ag.team != pteam:
                continue
            result[ag_id] = carrier_reward if ag_id == carrier_id else teammate_reward

        return result

    def _handle_goal_rewards(self, rewards, infos, scoring_team):
        """
        Ratio-chain investment payout at goal:

        Chain example after 2 passes:
          [(A, 0.25, s0), (B, 0.50, s1), (C, 1.00, s2)]  -- C is scorer

        - Scorer (last in chain): receives ratio * GOAL_REWARD immediately.
        - Passers (everyone else in chain): receive ratio * GOAL_REWARD
          retroactively patched to the step they entered the chain,
          via the existing callback mechanism (investment_credit).
        - If chain is empty (no touches recorded): scorer gets full GOAL_REWARD.
        """
        GOAL_REWARD = 10.0
        early_goal_bonus = 0.001 * max(0, self.max_steps - self.step_count)

        scorer_id = self._inv_last_toucher

        for ag_id, ag in self.agents.items():
            if ag.team == scoring_team:
                # Thưởng ghi bàn sớm cho tất cả agent thuộc đội ghi bàn
                rewards[ag_id] += early_goal_bonus
                
                # Find this agent's entry in the chain
                chain_entry = next(((r, s) for aid, r, s in self._inv_chain if aid == ag_id), None)

                if chain_entry is None:
                    # Agent never touched the ball this possession
                    if ag_id == scorer_id or len(self._inv_chain) == 0:
                        # Fallback: no chain at all, give full reward
                        rewards[ag_id] += GOAL_REWARD
                    # else: not in chain, no reward
                elif ag_id == scorer_id:
                    # Scorer: pay immediately
                    ratio, _ = chain_entry
                    rewards[ag_id] += ratio * GOAL_REWARD
                else:
                    # Passer: retroactive credit back to their entry step
                    ratio, entry_step = chain_entry
                    steps_ago = self._inv_step - entry_step
                    if steps_ago >= 0:
                        infos[ag_id]["investment_credit"] = [(steps_ago, ratio * GOAL_REWARD)]
            else:
                # Conceding penalty
                losing_team = ag.team
                last_toucher_id = self._inv_last_toucher
                last_toucher_team = self.agents[last_toucher_id].team if last_toucher_id else None

                if last_toucher_team == losing_team:
                    # Own goal: only the player who scored the own goal is penalized
                    if ag_id == last_toucher_id:
                        rewards[ag_id] -= 6.0
                    # else: 0.0 for other teammates
                else:
                    # Normal concede
                    rewards[ag_id] -= 3.0
                
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

        # Investment Update (ratio-chain)
        if touching_ag_ids:
            # If possession changed to a new team, record loss event then reset chain
            if self.possession_team != prev_pos and prev_pos != 0:
                self._possession_loss_event = (prev_pos, self._inv_last_toucher)
                self._inv_chain = []
                self._inv_last_toucher = None

            # Only track touches from the possessing team
            possessing_touchers = [aid for aid in touching_ag_ids
                                   if self.agents[aid].team == self.possession_team]
            for ag_id in possessing_touchers:
                if self._inv_last_toucher is None:
                    # First touch: start chain with ratio 1.0
                    self._inv_chain = [(ag_id, 1.0, self._inv_step)]
                    self._inv_last_toucher = ag_id
                elif self._inv_last_toucher != ag_id:
                    # New player entered → halve all existing ratios, append new entry at 1.0
                    self._inv_chain = [(aid, r * 0.5, s) for aid, r, s in self._inv_chain]
                    self._inv_chain.append((ag_id, 1.0, self._inv_step))
                    self._inv_last_toucher = ag_id
                # Same player touching again → no change

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

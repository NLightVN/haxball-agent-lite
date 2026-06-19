"""
env_a3.py — Phase A3: 3v3 Multi-Agent Shared Policy Environment
================================================================

Reward logic (2 loại thưởng):

POSSESSION
  last_touch_team     — công tắc đơn giản: phe nào chạm bóng cuối cùng.
  prev_poss_at_touch  — cập nhật DUY NHẤT tại thời điểm touch mới:
                        = last_touch_team cũ  nếu khác team mới VÀ dt <= 0.25s
                        = None               nếu cùng team hoặc bóng tự do > 0.25s

LOẠI 1 — BALL MOVEMENT REWARD (thưởng liên tục):
  Case 1: possession=team mình VÀ prev_poss_at_touch != opponent
          → Holder KHÔNG bị thưởng/phạt về di chuyển bóng.
            Tỉ lệ invest (từ trực tiếp đến xa): [1.0, 0.3, 0.15, 0.075, ...]
  Case 2: possession=team mình VÀ prev_poss_at_touch=opponent, HOẶC possession=opponent
          → Holder bị thưởng/phạt bình thường.
            Tỉ lệ invest (holder đứng đầu): [1.0, 1.0, 0.3, 0.15, 0.075, ...]

LOẠI 2 — PASS BONUS REWARD:
  Khi real_pass thành công (receiver != shooter):
    bonus = max(0, cumulative_ball_adv)  — tích lũy kể từ lần hold đầu
    Tỉ lệ invest cho các agent đã đầu tư (không phải receiver):
      [1.0, 0.3, 0.15, 0.075, 0.0375, ...] (receiver ăn phần tử đầu nếu trong seq)

INVESTMENT SEQUENCE  (N chuỗi chủ quan, N = số agent trong team)
  Bị xóa toàn bộ khi đối thủ giữ bóng liên tục >= 2 giây.

GOAL
  Ghi bàn:  +20.0 base  +3.0 bonus_pool (chia invest_share cho assisters)
  Bị ghi:   -20.0

TURNOVER PENALTY  = -0.1 (nếu không trong sequence)  /  -0.1 * invest_share (nếu trong sequence)
"""

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from training.env import (
    HaxballCurriculumEnv,
    Disc,
    _resolve_dd,
    DIR_MAP,
    BALL_R, BALL_DAMP, BALL_BCOEF, BALL_IMASS,
    PLYR_R, PLYR_DAMP, PLYR_IMASS, PLYR_BCOEF, PLYR_ACC, PLYR_KICK_ACC,
    KICK_STR, KICK_RANGE,
    POLE_R, POLE_BCOEF, POLE_IMASS,
    OUTER_PAD, GOAL_DEPTH,
    NORM, MAX_SPEED, DIAG,
    N_TM, N_OPP, OBS_DIM,
    MAP_PRESETS,
)

# ─── Constants ────────────────────────────────────────────────────────────────
FRAME_SKIP      = 3          # physics ticks per step
PHYSICS_HZ      = 60
MAX_STEPS       = 1800       # 90s / (3 ticks / 60Hz)

N_AGENTS        = 3          # agents per team
TOTAL_AGENTS    = 6          # 3 red + 3 blue

ADVANCE_REWARD   = 0.003
BACKWARD_PENALTY = 0.003

SELF_PASS_EXPIRE = 2.5       # seconds

BONUS_POOL       = 3.0
GOAL_REWARD      = 20.0
CONCEDE_PENALTY  = 20.0

TURNOVER_PEN     = -0.1

MAX_STEPS_ALL    = 6000      # for obs normalisation

# Investment share ratios
# Case 1 (holder excluded): direct investor gets 1.0, further: 0.3, 0.15, 0.075, ...
INVEST_CASE1 = [1.0, 0.3, 0.15, 0.075]
# Case 2 (holder included): holder gets 1.0, direct investor: 1.0, 0.3, 0.15, 0.075, ...
INVEST_CASE2 = [1.0, 1.0, 0.3, 0.15, 0.075]

# ─── Helper ───────────────────────────────────────────────────────────────────
def _predict_greedy_receiver(ball, agents, n_frames=150):
    """Predict which agent intercepts ball trajectory first (greedy)."""
    bx, by = ball.x, ball.y
    bxs, bys = ball.xs, ball.ys
    damp = ball.damp

    for t in range(1, n_frames + 1):
        bx  += bxs
        by  += bys
        bxs *= damp
        bys *= damp
        for ag in agents:
            if math.hypot(ag.x - bx, ag.y - by) <= ag.radius + ball.radius + 3.0 * t:
                return ag

    # Fallback: closest to final predicted position
    closest, min_d = None, float('inf')
    for ag in agents:
        d = math.hypot(ag.x - bx, ag.y - by)
        if d < min_d:
            min_d, closest = d, ag
    return closest


def _invest_share(seq, pid):
    """Return (invest_share, in_sequence) for agent pid given their sequence.
    Legacy: holder=1.0, -1=0.3, -2=0.15, ...
    """
    if not seq:
        return 0.0, False
    if seq[-1] == pid:
        return 1.0, True
    if pid in seq:
        idx = seq.index(pid)
        passes_away = len(seq) - 1 - idx
        return 0.3 * (0.5 ** (passes_away - 1)), True
    return 0.0, False


def _invest_share_case1(seq, pid):
    """Case 1: holder bị loại khỏi thưởng/phạt ball movement.
    Người trực tiếp invest vào holder (seq[-2]) ăn 1.0,
    tiếp theo: 0.3, 0.15, 0.075, ...
    Returns (share, in_sequence).
    """
    if not seq or len(seq) < 2:
        # Chỉ có 1 người trong seq (holder), không ai được chia
        return 0.0, False
    if seq[-1] == pid:
        # Holder → 0 (bị loại)
        return 0.0, True
    if pid not in seq:
        return 0.0, False
    # Vị trí tính từ cuối: seq[-2] → ratios[0]=1.0, seq[-3] → ratios[1]=0.3, ...
    idx = seq.index(pid)
    pos_from_end = len(seq) - 1 - idx  # 1-based: 1 = trực tiếp, 2 = 1 pass xa, ...
    if pos_from_end <= len(INVEST_CASE1):
        return INVEST_CASE1[pos_from_end - 1], True
    # Ngoài bảng: tiếp tục giảm theo hệ số 0.5
    return INVEST_CASE1[-1] * (0.5 ** (pos_from_end - len(INVEST_CASE1))), True


def _invest_share_case2(seq, pid):
    """Case 2: holder được bao gồm, ăn 1.0 cùng vị trí đầu.
    Holder (seq[-1]) = ratios[0]=1.0,
    direct investor (seq[-2]) = ratios[1]=1.0,
    tiếp theo: 0.3, 0.15, 0.075, ...
    Returns (share, in_sequence).
    """
    if not seq:
        return 0.0, False
    if pid not in seq:
        return 0.0, False
    idx = seq.index(pid)
    pos_from_end = len(seq) - 1 - idx  # 0 = holder, 1 = direct, ...
    if pos_from_end < len(INVEST_CASE2):
        return INVEST_CASE2[pos_from_end], True
    return INVEST_CASE2[-1] * (0.5 ** (pos_from_end - (len(INVEST_CASE2) - 1))), True



# ─── Environment ──────────────────────────────────────────────────────────────
class A3Env(gym.Env):
    """
    Phase A3 — 3v3 Multi-Agent Shared Policy (no opponent team, self-play ready).
    Action : MultiDiscrete([9, 2]) per agent.
    Obs    : OBS_DIM float32 per agent (same as single-agent env).
    step() : receives list of 3 actions, returns list of 3 obs/rewards.
    """

    metadata = {'render_modes': []}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()

        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([9, 2])

        # ── Field / physics state (set in reset) ──────────────────────────────
        self.ball:   Optional[Disc] = None
        self.agents: list           = []
        self.HW = self.HH = self.goal_y = self.goal_center_y = 0.0

        # ── Episode meta ──────────────────────────────────────────────────────
        self.phase          = 'A3'
        self.frame_skip     = FRAME_SKIP
        self.max_steps      = MAX_STEPS
        self.step_count     = 0
        self.scores         = [0, 0]
        self.team_id        = 1
        self._flip          = 1.0
        self._attack_sign   = 1
        self.total_timesteps_elapsed: int = 0

        # ── Possession state ───────────────────────────────────────────────────
        self.last_touch_team    = None   # int(team_id) or None
        self.last_touch_time    = -999.0
        self.prev_poss_at_touch = None   # opp_id or None (never team_id itself)

        # ── Investment sequences [seq_for_agent_0, seq_for_agent_1, seq_for_agent_2]
        self.invest_seqs: list = [[], [], []]
        self.opp_possession_time = 0.0

        # ── Self-pass / no-pass state ──────────────────────────────────────────
        # self_pass[i]: True if agent i last kicked and ball predicted back to themselves
        self.self_pass      = [False, False, False]
        self.self_pass_time = [0.0,   0.0,   0.0]   # time when self_pass was set
        # no_pass[i]: True if agent i is current holder but has NOT done a real_pass yet
        self.no_pass        = [True, True, True]

        # ── Step tracking ──────────────────────────────────────────────────────
        self._prev_ball_x = 0.0
        self._prev_ball_y = 0.0
        self.kick_exhausted = [False] * TOTAL_AGENTS
        self.touches_this_tick = []

        # Opponent support (for SharedPolicyVecEnv self-play)
        self.override_opponent_policy = None

    # ─── Reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0
        self.scores     = [0, 0]

        self.team_id      = int(self._rng.integers(1, 3))
        self._flip        = 1.0 if self.team_id == 1 else -1.0
        self._attack_sign = 1   if self.team_id == 1 else -1

        self._reset_positions()

        self.last_touch_team    = None
        self.last_touch_time    = -999.0
        self.prev_poss_at_touch = None

        for seq in self.invest_seqs:
            seq.clear()
        self.opp_possession_time = 0.0

        self.self_pass      = [False, False, False]
        self.self_pass_time = [0.0,   0.0,   0.0]
        self.no_pass        = [True, True, True]

        self._prev_ball_x = self.ball.x
        self._prev_ball_y = self.ball.y
        self.kick_exhausted = [False] * TOTAL_AGENTS

        return [self._get_obs(i) for i in range(N_AGENTS)], {}

    def _reset_positions(self):
        # Use 3v3 map presets
        cands = [p for p in MAP_PRESETS if p[3] == '3v3'] or MAP_PRESETS
        preset = cands[int(self._rng.integers(0, len(cands)))]
        self.HW, self.HH, self.goal_y = float(preset[0]), float(preset[1]), float(preset[2])
        self.goal_center_y = 0.0

        # Assign 3 different bot types for opponents (shuffled from pool each episode)
        _bot_pool = ['Wanderer', 'Random', 'Pazzo', 'Static']
        sampled = list(self._rng.choice(_bot_pool, size=N_AGENTS, replace=False))
        self._opp_types = sampled
        self._opp_policy = getattr(self, 'override_opponent_policy', None)

        # Spawn agents — red at x<0, blue at x>0
        self.agents = []
        for _ in range(N_AGENTS):   # red team
            x = float(self._rng.uniform(-self.HW + PLYR_R, -PLYR_R))
            y = float(self._rng.uniform(-self.HH + PLYR_R,  self.HH - PLYR_R))
            self.agents.append(Disc(x, y, 0, 0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))
        for _ in range(N_AGENTS):   # blue team
            x = float(self._rng.uniform(PLYR_R, self.HW - PLYR_R))
            y = float(self._rng.uniform(-self.HH + PLYR_R,  self.HH - PLYR_R))
            self.agents.append(Disc(x, y, 0, 0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP))

        # team_id=2 means blue plays from left (swap halves)
        if self.team_id == 2:
            self.agents = self.agents[N_AGENTS:] + self.agents[:N_AGENTS]

        # Ball at centre
        self.ball = Disc(0.0, 0.0, 0, 0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)

    # ─── Step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        action: list of 3 actions, each = [dir_idx, kick]
        Returns: obs_list, rew_list, terminated, truncated, info_list
        """
        assert self.ball is not None, "Call reset() first"

        # ── Build agent_actions [(dx,dy,kick), ...] for all 6 agents ──────────
        agent_actions = []
        for i in range(N_AGENTS):
            dir_idx = int(action[i][0])
            kick    = int(action[i][1])
            dx, dy  = DIR_MAP[dir_idx]
            agent_actions.append((dx * self._flip, dy, kick))

        # Opponent team actions (indices N_AGENTS..2*N_AGENTS-1)
        opp_id = 2 if self.team_id == 1 else 1
        for opp_i in range(N_AGENTS):
            ag = self.agents[N_AGENTS + opp_i]
            if self._opp_policy is not None:
                obs_opp = self._get_obs_for_opponent(opp_i)
                opp_act, _ = self._opp_policy.predict(obs_opp, deterministic=False)
                opp_dx, opp_dy = DIR_MAP[int(opp_act[0])]
                agent_actions.append((opp_dx * -self._flip, opp_dy, int(opp_act[1])))
            else:
                bot_type = self._opp_types[opp_i]
                if bot_type == 'Follower':
                    agent_actions.append(self._get_follower_action(ag))
                elif bot_type == 'Random':
                    agent_actions.append(self._get_random_action())
                elif bot_type == 'Pazzo':
                    agent_actions.append(self._get_pazzo_action(ag, opp_i))
                elif bot_type == 'Static':
                    agent_actions.append((0.0, 0.0, 0))
                else:  # Wanderer
                    agent_actions.append(self._get_wanderer_action(ag, opp_i))

        # ── Save previous ball position ────────────────────────────────────────
        px, py = self._prev_ball_x, self._prev_ball_y

        # ── Physics ticks ──────────────────────────────────────────────────────
        self.kick_exhausted = [False] * TOTAL_AGENTS
        goal_result = 0
        touch_events = []   # list of (agent_idx, team_id)

        for _ in range(FRAME_SKIP):
            self.touches_this_tick = []
            result = self._tick(agent_actions)
            touch_events.extend(self.touches_this_tick)
            if result != 0:
                goal_result = result
                break

        self.step_count += 1
        time_now = self.step_count * (FRAME_SKIP / PHYSICS_HZ)

        # ── Process touch events ───────────────────────────────────────────────
        turnover = False
        for (pid, tid) in touch_events:
            dt = time_now - self.last_touch_time

            # Update prev_poss_at_touch (frozen snapshot at moment of touch)
            if (self.last_touch_team is not None
                    and self.last_touch_team != tid
                    and dt <= 0.25):
                self.prev_poss_at_touch = self.last_touch_team
            else:
                self.prev_poss_at_touch = None

            # Turnover detection
            if self.last_touch_team == self.team_id and tid == opp_id:
                turnover = True

            self.last_touch_team = tid
            self.last_touch_time = time_now

            # Investment sequences — only update when our team touches
            if tid == self.team_id:
                for j in range(N_AGENTS):
                    if j == pid:
                        # Self-touch: reset own sequence to [pid]
                        self.invest_seqs[j] = [pid]
                    else:
                        # Teammate touched: append pid (remove duplicate first)
                        seq = self.invest_seqs[j]
                        if pid in seq:
                            seq.remove(pid)
                        seq.append(pid)

                # Kick was made → check self_pass / real_pass
                # (we check per agent based on who just touched)
                if pid < N_AGENTS:
                    # Determine if this agent used kick action this step
                    kicked_this_step = (agent_actions[pid][2] == 1)
                    if kicked_this_step:
                        receiver = _predict_greedy_receiver(self.ball, self.agents[:N_AGENTS])
                        if receiver is self.agents[pid]:
                            # Self-pass: sút về mình
                            self.self_pass[pid]      = True
                            self.self_pass_time[pid] = time_now
                            self.no_pass[pid]        = True   # no real pass
                        elif receiver is not None and receiver in self.agents[:N_AGENTS]:
                            # Real pass to a different teammate
                            self.self_pass[pid]  = False
                            self.no_pass[pid]    = False  # cleared!
                        else:
                            # Kicked to nobody/opponent
                            self.self_pass[pid] = False
                            self.no_pass[pid]   = True
                    else:
                        # Touched without kick → still no_pass
                        self.self_pass[pid] = False
                        self.no_pass[pid]   = True

        # Expire self_pass after 2.5s
        for i in range(N_AGENTS):
            if self.self_pass[i] and (time_now - self.self_pass_time[i]) > SELF_PASS_EXPIRE:
                self.self_pass[i] = False

        # Opponent possession time (for investment sequence reset)
        if self.last_touch_team == opp_id:
            self.opp_possession_time += FRAME_SKIP / PHYSICS_HZ
            if self.opp_possession_time >= 2.0:
                for seq in self.invest_seqs:
                    seq.clear()
        else:
            self.opp_possession_time = 0.0

        # ── Possession flags ───────────────────────────────────────────────────
        prev_poss = self.prev_poss_at_touch   # opp_id or None

        # Case 1: team mình đang cầm VÀ prev không phải opponent (bóng liên tục của mình)
        #   → holder bị loại khỏi thưởng/phạt ball movement
        # Case 2: team mình cầm nhưng prev=opp, HOẶC opponent cầm
        #   → holder được bao gồm
        our_poss_continuous = (
            self.last_touch_team == self.team_id
            and prev_poss != opp_id   # prev_poss is None (liên tục ta) hoặc = team_id
        )
        is_case1 = our_poss_continuous
        is_case2 = (
            (self.last_touch_team == self.team_id and prev_poss == opp_id)
            or (self.last_touch_team == opp_id)
        )

        # is_opp_pass_back: đối thủ tự pass lùi → mult = 0 (giữ nguyên logic cũ)
        is_opp_pass_back = (prev_poss == opp_id and self.last_touch_team == opp_id)

        # ── Delta distance to goal ─────────────────────────────────────────────
        bx, by = self.ball.x, self.ball.y
        atk    = self._attack_sign
        adv_x  = bx * atk
        zone_w = self.goal_y * 2.0

        if adv_x <= -self.HW + zone_w:
            own_gx  = -self.HW * atk
            delta   = math.hypot(bx - own_gx, by) - math.hypot(px - own_gx, py)
        elif adv_x >= self.HW - zone_w:
            opp_gx  = self.HW * atk
            delta   = math.hypot(px - opp_gx, py) - math.hypot(bx - opp_gx, by)
        else:
            delta   = (bx - px) * atk

        # ── Goal events ────────────────────────────────────────────────────────
        terminated = False
        truncated  = False
        base_rew   = [0.0] * N_AGENTS

        if goal_result == 2:          # we scored
            self.scores[self.team_id - 1] += 1
            for i in range(N_AGENTS):
                base_rew[i] += GOAL_REWARD
            # Assister bonus
            for i in range(N_AGENTS):
                share, in_seq = _invest_share(self.invest_seqs[i], i)
                if in_seq and self.invest_seqs[i][-1] != i:
                    base_rew[i] += BONUS_POOL * share
            terminated = True

        elif goal_result == 1:        # opponent scored
            self.scores[opp_id - 1] += 1
            for i in range(N_AGENTS):
                base_rew[i] -= CONCEDE_PENALTY
            terminated = True

        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        # ── Per-agent dense rewards ────────────────────────────────────────────
        rew_list = list(base_rew)
        info_adv    = [0.0] * N_AGENTS
        info_back   = [0.0] * N_AGENTS
        info_turn   = [0.0] * N_AGENTS
        info_sp     = list(self.self_pass)
        info_np     = list(self.no_pass)
        info_case   = [0] * N_AGENTS  # 1 or 2 for debug

        for i in range(N_AGENTS):
            seq_i = self.invest_seqs[i]

            # ── Loại 1: Ball movement reward ──────────────────────────────────
            if is_case1:
                # Case 1: holder nhận thưởng khi bóng tiến.
                # Holder KHÔNG bị phạt khi lùi — TRỪ KHI đang dribble (no_pass=True):
                #   dribble = chạm bóng nhưng chưa real_pass (vật lý, self_pass, sút vào ai đó)
                share1, in_seq1 = _invest_share_case1(seq_i, i)
                info_case[i] = 1

                is_holder = bool(seq_i) and seq_i[-1] == i
                # Dribble: holder đang cầm bóng mà chưa chuyền cho đồng đội
                holder_dribble = is_holder and self.no_pass[i]

                if delta > 0:
                    # Holder được thưởng bình thường
                    if is_holder:
                        adv_val = ADVANCE_REWARD * delta * 1.0
                    else:
                        adv_val = ADVANCE_REWARD * delta * share1
                    rew_list[i] += adv_val
                    info_adv[i]  = adv_val
                elif delta < 0:
                    if is_holder and not holder_dribble:
                        # Holder đã real_pass (no_pass=False) → không phạt
                        pass
                    elif is_holder and holder_dribble:
                        # Dribble → phạt full 1.0
                        back_val = BACKWARD_PENALTY * abs(delta) * 1.0
                        rew_list[i] -= back_val
                        info_back[i] = -back_val
                    else:
                        # Investor bị phạt bình thường
                        back_val = BACKWARD_PENALTY * abs(delta) * share1
                        rew_list[i] -= back_val
                        info_back[i] = -back_val

            elif is_case2:
                # Case 2: holder được bao gồm, tỉ lệ INVEST_CASE2
                # is_opp_pass_back → opponent tự pass lùi: chỉ phạt (không thưởng)
                share2, in_seq2 = _invest_share_case2(seq_i, i)
                info_case[i] = 2

                if delta > 0 and not is_opp_pass_back:
                    adv_val = ADVANCE_REWARD * delta * share2
                    rew_list[i] += adv_val
                    info_adv[i]  = adv_val
                elif delta < 0:
                    # Opponent cầm bóng: luôn phạt dù is_opp_pass_back hay không
                    back_val = BACKWARD_PENALTY * abs(delta) * share2
                    rew_list[i] -= back_val
                    info_back[i] = -back_val

            # ── Turnover penalty ────────────────────────────────────────────────
            if turnover:
                # Dùng legacy share để tính turnover penalty
                share_leg, in_seq_leg = _invest_share(seq_i, i)
                turn_val = TURNOVER_PEN * (share_leg if in_seq_leg else 1.0)
                rew_list[i] += turn_val
                info_turn[i] = turn_val

        # ── Update tracking ────────────────────────────────────────────────────
        self._prev_ball_x = self.ball.x
        self._prev_ball_y = self.ball.y

        info = {
            "marl/self_pass":    info_sp,
            "marl/no_pass":      info_np,
            "marl/adv_rew":      info_adv,
            "marl/back_pen":     info_back,
            "marl/turn_pen":     info_turn,
            "marl/reward_case":  info_case,
            "marl/opp_pos_time": float(self.opp_possession_time),
            "marl/invest_seqs":  [list(s) for s in self.invest_seqs],
        }

        obs_list = [self._get_obs(i) for i in range(N_AGENTS)]
        return obs_list, rew_list, terminated, truncated, [info] * N_AGENTS

    # ─── Physics ──────────────────────────────────────────────────────────────
    def _tick(self, agent_actions) -> int:
        """One physics tick. Returns 0=normal, 1=left-goal, 2=right-goal."""
        ball   = self.ball
        agents = self.agents
        HW, HH = self.HW, self.HH

        action_order = list(range(TOTAL_AGENTS))
        self._rng.shuffle(action_order)

        # Mask kick: only accept if in range AND not exhausted
        masked = []
        for i in range(TOTAL_AGENTS):
            dx, dy, kick = agent_actions[i]
            if kick == 0:
                self.kick_exhausted[i] = False
                masked.append((dx, dy, 0))
            else:
                d = math.hypot(ball.x - agents[i].x, ball.y - agents[i].y)
                in_range = d - agents[i].radius - ball.radius < KICK_RANGE
                if in_range and not self.kick_exhausted[i]:
                    masked.append((dx, dy, 1))
                else:
                    masked.append((dx, dy, 0))

        # 1. Kick
        for i in action_order:
            ag = agents[i]
            dx, dy, kick = masked[i]
            if kick:
                dx_b = ball.x - ag.x
                dy_b = ball.y - ag.y
                dist = math.hypot(dx_b, dy_b)
                if dist > 0 and dist - ag.radius - ball.radius < KICK_RANGE:
                    nx, ny = dx_b / dist, dy_b / dist
                    ball.xs += nx * KICK_STR
                    ball.ys += ny * KICK_STR
                    self.kick_exhausted[i] = True
                    team = self.team_id if i < N_AGENTS else (2 if self.team_id == 1 else 1)
                    self.touches_this_tick.append((i, team))

        # 2. Acceleration
        for i in action_order:
            ag = agents[i]
            dx, dy, _ = masked[i]
            intent_kick = agent_actions[i][2]
            ln = math.hypot(dx, dy)
            acc = PLYR_KICK_ACC if intent_kick else PLYR_ACC
            if ln > 0:
                ag.xs += (dx / ln) * acc
                ag.ys += (dy / ln) * acc

        # 3. Move
        ball.x += ball.xs; ball.y += ball.ys
        for ag in agents:
            ag.x += ag.xs; ag.y += ag.ys

        # 4. Disc-disc collisions
        shuffled = list(agents)
        self._rng.shuffle(shuffled)
        all_discs = [ball] + shuffled
        n = len(all_discs)
        for i in range(n):
            for j in range(i + 1, n):
                if _resolve_dd(all_discs[i], all_discs[j]):
                    if i == 0 or j == 0:
                        ag_disc = all_discs[max(i, j)]
                        try:
                            orig = agents.index(ag_disc)
                            team = self.team_id if orig < N_AGENTS else (2 if self.team_id == 1 else 1)
                            self.touches_this_tick.append((orig, team))
                        except ValueError:
                            pass

        # 4b. Pole collisions
        poles = [
            Disc( HW, self.goal_center_y + self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
            Disc( HW, self.goal_center_y - self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
            Disc(-HW, self.goal_center_y + self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
            Disc(-HW, self.goal_center_y - self.goal_y, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),
        ]
        for pole in poles:
            for ag in agents:
                _resolve_dd(ag, pole)
            _resolve_dd(ball, pole)

        # 5. Wall collisions — Ball
        if ball.y - ball.radius < -HH:
            ball.y = -HH + ball.radius; ball.ys = -ball.ys * ball.bcoef
        if ball.y + ball.radius >  HH:
            ball.y =  HH - ball.radius; ball.ys = -ball.ys * ball.bcoef

        atk = self._attack_sign
        if ball.x - ball.radius < -HW:
            if self.goal_center_y - self.goal_y < ball.y < self.goal_center_y + self.goal_y:
                return 1 if atk == 1 else 2
            ball.x = -HW + ball.radius; ball.xs = -ball.xs * ball.bcoef
        elif ball.x + ball.radius > HW:
            if self.goal_center_y - self.goal_y < ball.y < self.goal_center_y + self.goal_y:
                return 2 if atk == 1 else 1
            ball.x =  HW - ball.radius; ball.xs = -ball.xs * ball.bcoef

        # Wall collisions — Players
        max_y = HH + OUTER_PAD
        max_x = HW + GOAL_DEPTH
        for ag in agents:
            if ag.x - ag.radius < -max_x: ag.x = -max_x + ag.radius; ag.xs = max(ag.xs, 0)
            if ag.x + ag.radius >  max_x: ag.x =  max_x - ag.radius; ag.xs = min(ag.xs, 0)
            if ag.y - ag.radius < -max_y: ag.y = -max_y + ag.radius; ag.ys = max(ag.ys, 0)
            if ag.y + ag.radius >  max_y: ag.y =  max_y - ag.radius; ag.ys = min(ag.ys, 0)

        # 6. Damping
        ball.xs *= ball.damp; ball.ys *= ball.damp
        for ag in agents:
            ag.xs *= ag.damp; ag.ys *= ag.damp

        return 0

    # ─── Observation ──────────────────────────────────────────────────────────
    def _get_obs(self, agent_idx: int) -> np.ndarray:
        """Build OBS_DIM observation for agent agent_idx (0..2 = our team)."""
        ball  = self.ball
        agent = self.agents[agent_idx]
        obs   = np.zeros(OBS_DIM, dtype=np.float32)

        bx, by   = ball.x,  ball.y
        bxs, bys = ball.xs, ball.ys
        mx, my   = agent.x, agent.y
        mxs, mys = agent.xs, agent.ys
        flip     = self._flip

        surf_dist_raw = math.hypot(mx - bx, my - by) - PLYR_R - BALL_R
        surf_dist     = max(0.0, surf_dist_raw)
        dist_diff     = surf_dist_raw - KICK_RANGE
        my_dist_ball  = math.hypot(mx - bx, my - by)

        i = 0

        # Section 1 — Field constants (4)
        obs[i] = self.goal_y / NORM;      i += 1
        obs[i] = self.HH / NORM;          i += 1
        obs[i] = self.HW / NORM;          i += 1
        obs[i] = 0.0;                     i += 1   # team colour (always 0=RED from agent's POV)

        # Section 2 — Agent ↔ Ball (7)
        obs[i] = flip * (bx - mx) / NORM; i += 1
        obs[i] = (by - my) / NORM;        i += 1
        obs[i] = surf_dist / DIAG;        i += 1
        obs[i] = dist_diff / DIAG;        i += 1

        is_nearest_field = 1.0
        for ag in self.agents:
            if ag is not agent and math.hypot(ag.x - bx, ag.y - by) < my_dist_ball:
                is_nearest_field = 0.0; break
        obs[i] = is_nearest_field; i += 1

        team_dists = sorted(
            math.hypot(self.agents[j].x - bx, self.agents[j].y - by)
            for j in range(N_AGENTS)
        )
        is_nearest_team = 1.0 if team_dists and team_dists[0] >= my_dist_ball - 0.001 else 0.0
        obs[i] = is_nearest_team; i += 1
        rank = sum(1.0 for d in team_dists if d < my_dist_ball - 0.001)
        obs[i] = rank; i += 1

        # Section 2b — Ball ↔ Goals (8)
        top_post_y = self.goal_center_y - self.goal_y
        bot_post_y = self.goal_center_y + self.goal_y

        obs[i] = (self.HW - flip * bx) / NORM; i += 1
        obs[i] = (top_post_y - by) / NORM;     i += 1
        obs[i] = (self.HW - flip * bx) / NORM; i += 1

        # invest_share for this agent (in their own sequence)
        share_obs, _ = _invest_share(self.invest_seqs[agent_idx], agent_idx)
        obs[i] = share_obs; i += 1

        obs[i] = (bot_post_y - by) / NORM;     i += 1
        obs[i] = (-self.HW - flip * bx) / NORM; i += 1
        obs[i] = (top_post_y - by) / NORM;     i += 1
        obs[i] = (-self.HW - flip * bx) / NORM; i += 1
        obs[i] = (bot_post_y - by) / NORM;     i += 1

        # Section 3 — Dynamic state (11)
        obs[i] = flip * bx / NORM;                        i += 1
        obs[i] = (by - self.goal_center_y) / NORM;        i += 1
        obs[i] = flip * bxs / MAX_SPEED;                  i += 1
        obs[i] = bys / MAX_SPEED;                         i += 1
        obs[i] = flip * mx / NORM;                        i += 1
        obs[i] = (my - self.goal_center_y) / NORM;        i += 1
        obs[i] = flip * mxs / MAX_SPEED;                  i += 1
        obs[i] = mys / MAX_SPEED;                         i += 1
        obs[i] = math.hypot(mxs, mys) / MAX_SPEED;        i += 1
        obs[i] = flip * (mxs - bxs) / MAX_SPEED;          i += 1
        obs[i] = (mys - bys) / MAX_SPEED;                 i += 1

        # Section 4 — Game state (2)
        obs[i] = max(0.0, 1.0 - self.step_count / self.max_steps); i += 1
        obs[i] = self.max_steps / MAX_STEPS_ALL;                    i += 1

        # Section 5 — MARL Possession (12)
        opp_id = 2 if self.team_id == 1 else 1

        # Possession current (3)
        if self.last_touch_team is None:       obs[i:i+3] = [1, 0, 0]
        elif self.last_touch_team == self.team_id: obs[i:i+3] = [0, 1, 0]
        else:                                  obs[i:i+3] = [0, 0, 1]
        i += 3

        # Previous possession 0.25s (3)
        pp = self.prev_poss_at_touch
        if pp is None:                         obs[i:i+3] = [1, 0, 0]
        elif pp == self.team_id:               obs[i:i+3] = [0, 1, 0]
        else:                                  obs[i:i+3] = [0, 0, 1]
        i += 3

        # Opp possession time (1)
        obs[i] = min(1.0, self.opp_possession_time / 2.0); i += 1

        # invest_share (1)
        obs[i] = share_obs; i += 1

        # Tangent vectors helper
        def _tangent(px, py):
            dx = px - bx; dy = py - by
            d2 = dx*dx + dy*dy
            R  = PLYR_R + BALL_R
            if d2 <= R*R:
                return 0.0, 0.0, 0.0, 0.0
            d  = math.sqrt(d2)
            L  = math.sqrt(d2 - R*R)
            th = math.atan2(dx, -dy)
            al = math.asin(R / d)
            a1 = th + al; a2 = th - al
            return flip*L*math.sin(a1)/NORM, L*(-math.cos(a1))/NORM, \
                   flip*L*math.sin(a2)/NORM, L*(-math.cos(a2))/NORM

        # Main agent tangent vectors (4)
        obs[i:i+4] = _tangent(mx, my); i += 4

        # Teammates (N_TM slots × 14)
        tm_count = 0
        for j in range(N_AGENTS):
            if j == agent_idx: continue
            if tm_count >= N_TM: break
            tm   = self.agents[j]
            idx2 = i + tm_count * 14

            obs[idx2]   = flip * tm.x / NORM;          idx2 += 1
            obs[idx2]   = tm.y / NORM;                  idx2 += 1
            obs[idx2]   = flip * tm.xs / MAX_SPEED;     idx2 += 1
            obs[idx2]   = tm.ys / MAX_SPEED;             idx2 += 1
            obs[idx2]   = flip * (tm.x - mx) / NORM;    idx2 += 1
            obs[idx2]   = (tm.y - my) / NORM;            idx2 += 1
            obs[idx2]   = flip * (bx - tm.x) / NORM;    idx2 += 1
            obs[idx2]   = (by - tm.y) / NORM;            idx2 += 1
            obs[idx2]   = max(0.0, math.hypot(tm.x-bx, tm.y-by)-PLYR_R-BALL_R) / DIAG; idx2 += 1
            t1x,t1y,t2x,t2y = _tangent(tm.x, tm.y)
            obs[idx2:idx2+4] = [t1x, t1y, t2x, t2y]; idx2 += 4
            tm_share, _ = _invest_share(self.invest_seqs[agent_idx], j)
            obs[idx2] = tm_share; idx2 += 1
            tm_count += 1

        i += N_TM * 14

        # Opponents (N_OPP slots × 13)
        opp_count = 0
        for j in range(N_AGENTS, TOTAL_AGENTS):
            if opp_count >= N_OPP: break
            opp  = self.agents[j]
            idx2 = i + opp_count * 13

            obs[idx2]   = flip * opp.x / NORM;           idx2 += 1
            obs[idx2]   = opp.y / NORM;                   idx2 += 1
            obs[idx2]   = flip * opp.xs / MAX_SPEED;      idx2 += 1
            obs[idx2]   = opp.ys / MAX_SPEED;              idx2 += 1
            obs[idx2]   = flip * (opp.x - mx) / NORM;     idx2 += 1
            obs[idx2]   = (opp.y - my) / NORM;             idx2 += 1
            obs[idx2]   = flip * (bx - opp.x) / NORM;     idx2 += 1
            obs[idx2]   = (by - opp.y) / NORM;             idx2 += 1
            obs[idx2]   = max(0.0, math.hypot(opp.x-bx, opp.y-by)-PLYR_R-BALL_R) / DIAG; idx2 += 1
            t1x,t1y,t2x,t2y = _tangent(opp.x, opp.y)
            obs[idx2:idx2+4] = [t1x, t1y, t2x, t2y]; idx2 += 4
            opp_count += 1

        i += N_OPP * 13

        return obs

    # ─── Opponent obs helper (for self-play) ──────────────────────────────────
    def _get_obs_for_opponent(self, opp_i: int) -> np.ndarray:
        """Return observation from the opponent's perspective (swap team context)."""
        orig_team = self.team_id
        orig_flip = self._flip
        orig_atk  = self._attack_sign

        self.team_id      = 2 if self.team_id == 1 else 1
        self._flip        = 1.0 if self.team_id == 1 else -1.0
        self._attack_sign = 1   if self.team_id == 1 else -1

        # Swap agent halves
        team1 = self.agents[:N_AGENTS]
        team2 = self.agents[N_AGENTS:TOTAL_AGENTS]
        self.agents = team2 + team1

        obs = self._get_obs(opp_i)

        # Revert
        self.agents       = team1 + team2
        self.team_id      = orig_team
        self._flip        = orig_flip
        self._attack_sign = orig_atk
        return obs

    # ─── Bot actions for opponent ──────────────────────────────────────────────
    def _get_wanderer_action(self, ag, opp_i):
        b = self.ball
        dist_to_ball = math.hypot(b.x - ag.x, b.y - ag.y)
        in_range = dist_to_ball - ag.radius - b.radius < KICK_RANGE

        if not hasattr(self, '_wand_steps'):
            self._wand_steps = {}; self._wand_interval = {}; self._wand_target = {}
        if opp_i not in self._wand_steps:
            self._wand_steps[opp_i] = 0
            self._wand_interval[opp_i] = int(self._rng.integers(1, 4))
            self._wand_target[opp_i] = (b.x, b.y)

        self._wand_steps[opp_i] += 1
        if self._wand_steps[opp_i] >= self._wand_interval[opp_i]:
            angle = float(self._rng.uniform(0, 2 * math.pi))
            r     = float(self._rng.uniform(0, 60.0))
            tx = max(-self.HW + PLYR_R, min(self.HW - PLYR_R, b.x + r * math.cos(angle)))
            ty = max(-self.HH + PLYR_R, min(self.HH - PLYR_R, b.y + r * math.sin(angle)))
            self._wand_target[opp_i] = (tx, ty)
            self._wand_steps[opp_i] = 0
            self._wand_interval[opp_i] = int(self._rng.integers(1, 4))

        tx, ty = self._wand_target[opp_i]
        dx, dy = tx - ag.x, ty - ag.y
        dist   = math.hypot(dx, dy)
        if dist < 1.0:
            hor, ver = 0.0, 0.0
        else:
            dx /= dist; dy /= dist
            best_idx, best_dot = 0, -2.0
            for k, (mx, my) in enumerate(DIR_MAP):
                dot = dx * mx + dy * my
                if dot > best_dot:
                    best_dot = dot; best_idx = k
            hor, ver = DIR_MAP[best_idx]

        kick = 1 if in_range and self._rng.random() < 0.10 else 0
        return (hor, ver, kick)

    def _get_follower_action(self, ag):
        """Chase ball, kick toward opponent goal. Simple anti-own-goal ray test."""
        b = self.ball
        # Opponent's own goal (follower defends from our side; they attack our goal)
        # Follower is opp → their goal = our goal = -self.HW * atk
        own_goal_x   = self.HW * self._attack_sign   # our goal (follower attacks this)
        own_goal_top = self.goal_center_y + self.goal_y
        own_goal_bot = self.goal_center_y - self.goal_y

        dist_to_ball = math.hypot(b.x - ag.x, b.y - ag.y)
        in_range = dist_to_ball - ag.radius - b.radius < KICK_RANGE

        would_own_goal = False
        if in_range:
            kx = b.x - ag.x; ky = b.y - ag.y
            kd = math.hypot(kx, ky)
            if kd > 0:
                nx, ny = kx / kd, ky / kd
                vx = b.xs + nx * KICK_STR
                vy = b.ys + ny * KICK_STR
                if abs(vx) > 0.01:
                    t = (own_goal_x - b.x) / vx
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
            target_x = b.x; target_y = b.y
            kick = 1 if in_range else 0

        dx, dy = target_x - ag.x, target_y - ag.y
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            return (0.0, 0.0, kick)
        dx /= dist; dy /= dist
        best_idx, best_dot = 0, -2.0
        for k, (mx, my) in enumerate(DIR_MAP):
            dot = dx * mx + dy * my
            if dot > best_dot:
                best_dot = dot; best_idx = k
        out_dx, out_dy = DIR_MAP[best_idx]
        return (out_dx, out_dy, kick)

    def _get_random_action(self):
        """Random direction + random kick."""
        dir_idx = int(self._rng.integers(0, 9))
        kick    = int(self._rng.integers(0, 2))
        dx, dy  = DIR_MAP[dir_idx]
        return (dx, dy, kick)

    def _get_pazzo_action(self, ag, opp_i):
        """Moves to a random waypoint, changes every 100-150 steps."""
        b = self.ball
        if not hasattr(self, '_pazzo_steps'):
            self._pazzo_steps = {}; self._pazzo_target = {}; self._pazzo_interval = {}
        if opp_i not in self._pazzo_steps:
            self._pazzo_steps[opp_i] = 0
            self._pazzo_target[opp_i] = (float(self._rng.uniform(-self.HW, self.HW)),
                                         float(self._rng.uniform(-self.HH, self.HH)))
            self._pazzo_interval[opp_i] = int(self._rng.integers(100, 151))

        self._pazzo_steps[opp_i] += 1
        if self._pazzo_steps[opp_i] > self._pazzo_interval[opp_i]:
            self._pazzo_target[opp_i] = (float(self._rng.uniform(-self.HW, self.HW)),
                                         float(self._rng.uniform(-self.HH, self.HH)))
            self._pazzo_steps[opp_i] = 0
            self._pazzo_interval[opp_i] = int(self._rng.integers(100, 151))

        if abs(b.x) < 5.0 and abs(b.y) < 5.0 and math.hypot(b.xs, b.ys) < 0.5:
            tx = float(self._rng.uniform(-150.0, 150.0))
            ty = float(self._rng.uniform(-150.0, 150.0))
        else:
            tx, ty = self._pazzo_target[opp_i]

        comp_x = tx - ag.x; comp_y = ty - ag.y
        denom = math.hypot(comp_x, comp_y)
        hor, ver = 0.0, 0.0
        if denom > 40.0:
            px_prob = abs(comp_x) / denom; py_prob = abs(comp_y) / denom
            if px_prob > py_prob: px_prob = 1.0
            elif py_prob > px_prob: py_prob = 1.0
            if self._rng.random() < px_prob: hor = 1.0 if comp_x > 0 else -1.0
            if self._rng.random() < py_prob: ver = 1.0 if comp_y > 0 else -1.0
        return (hor, ver, 0)

    # ─── Gymnasium boilerplate ────────────────────────────────────────────────
    def render(self):
        pass

    def action_masks(self) -> np.ndarray:
        """Mask kick action if out of range (for agent 0, used in play_a3.py)."""
        agent = self.agents[0]
        d = math.hypot(agent.x - self.ball.x, agent.y - self.ball.y)
        can_kick = (d - PLYR_R - BALL_R - KICK_RANGE) <= 0
        mask = np.ones(11, dtype=bool)
        mask[10] = can_kick
        return mask

    def close(self):
        pass

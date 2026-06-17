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

        for pid, tid in touch_events:
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
            else:
                # Teammate (or self) touched
                self.self_pass_active = False # Hit a teammate
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

        # Anti-self pass detection on KICK
        if kick and self.last_touch_team == self.team_id:
            # Simple raycast logic (for now, just flag if agent kicks and no one else is around)
            # A full raycast to find the closest intercepting teammate/opponent is complex.
            # We will approximate: if agent kicks, mark self_pass_active. It clears on ANY other touch.
            self.self_pass_active = True

        # ── Reward ────────────────────────────────────────────────────────────
        reward = 0.0
        terminated = False
        truncated = False

        atk = self._attack_sign
        goal_x = self.HW * atk

        # Compute Zones for X-axis
        # zone 1: Teammate (from -goal_x to x1)
        # zone 2: Mid (from x1 to x2)
        # zone 3: Opponent (from x2 to goal_x)
        # Let's say x1 = -0.33 * goal_x, x2 = 0.33 * goal_x
        x1 = -0.33 * goal_x
        x2 = 0.33 * goal_x
        
        # Calculate Delta X or Delta Dist
        cur_ball_dist_to_goal = _dist_to_goal_segment(self.ball.x, self.ball.y, goal_x, self.goal_y, -self.goal_y)
        delta_dist_to_goal = self._prev_ball_dist_to_goal - cur_ball_dist_to_goal
        
        # Ball Movement Reward Rules
        # + if (current_pos == self or pos_0_25 == self) and moving forward
        # - if (current_pos == opp or pos_0_25 == self) and moving backward
        has_possession_reward = (self.last_touch_team == self.team_id) or (possession_0_25s == self.team_id)
        has_possession_penalty = (self.last_touch_team == opp_id) or (possession_0_25s == self.team_id)
        
        ADVANCE_REWARD = 0.003
        BACKWARD_PENALTY = 0.001
        
        # Exception: Pass back to teammate -> 1/3 penalty
        is_pass_back = (possession_0_25s == self.team_id and self.last_touch_team == self.team_id)
        
        # Exception: Opp pass back to opp -> no reward
        is_opp_pass_back = (possession_0_25s == opp_id and self.last_touch_team == opp_id)
        
        dribble_duration = time_now - self.dribble_start_time
        is_dribbling = (dribble_duration > 1.0) # threshold for "farm dribble"
        
        if delta_dist_to_goal > 0: # Ball advanced (good)
            if has_possession_reward and not is_opp_pass_back:
                # Calculate multiplier
                mult = 1.0
                if is_dribbling and not kick:
                    mult = 0.333
                elif self.self_pass_active:
                    mult = 0.333
                reward += ADVANCE_REWARD * delta_dist_to_goal * mult
        elif delta_dist_to_goal < 0: # Ball moved back (bad)
            if has_possession_penalty:
                mult = 1.0
                if is_pass_back:
                    mult = 0.333
                reward -= BACKWARD_PENALTY * abs(delta_dist_to_goal) * mult

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
        self._prev_ball_speed   = math.hypot(self.ball.xs, self.ball.ys)

        info = {
            "marl/sequence_len": len(self.investment_sequence),
            "marl/dribble_duration": float(dribble_duration),
            "marl/self_pass": int(self.self_pass_active),
            "marl/opp_pos_time": float(self.opp_possession_time),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

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
        
        for tm_i in range(N_TM):
            # Currently 0 teammates
            idx = i + tm_i * 13 # 9 base + 4 tangent
            pass
        i += N_TM * 13

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

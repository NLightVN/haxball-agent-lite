// Agent API - Wrapper to access game state and control bots
// Built on top of Wazarr94's Haxball clone physics engine

// Helper: Compute bounding box (W x H) from the playable area segments
// Uses ballArea segments (cMask includes 'ball') vertex min/max
function getBoundingBox() {
    if (!stadium || !stadium.vertexes || !stadium.segments) {
        return { W: stadium.width * 2, H: stadium.height * 2 };
    }
    // Collect all vertex indices used by segments that affect the ball (ballArea trait)
    let xs = [], ys = [];
    stadium.segments.forEach(s => {
        const trait = s.trait ? (stadium.traits ? stadium.traits[s.trait] : null) : null;
        const isBallWall = trait && trait.cMask &&
            (Array.isArray(trait.cMask)
                ? trait.cMask.includes('ball')
                : (trait.cMask & 1) !== 0);  // bit 1 = ball flag
        if (isBallWall && s.v0 !== undefined && s.v1 !== undefined) {
            const v0 = stadium.vertexes[s.v0];
            const v1 = stadium.vertexes[s.v1];
            if (v0) { xs.push(v0.x); ys.push(v0.y); }
            if (v1) { xs.push(v1.x); ys.push(v1.y); }
        }
    });
    // Also include planes to bound the field
    if (xs.length === 0) {
        return { W: stadium.width * 2, H: stadium.height * 2 };
    }
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    let W = maxX - minX;
    let H = maxY - minY;

    // Condition: W > H (swap if necessary, though unlikely for Haxball)
    if (H > W) {
        [W, H] = [H, W];
    }

    return {
        W: W,
        H: H,
        minX, maxX, minY, maxY
    };
}

window.AgentAPI = {
    // Get current game state
    getState: function () {
        return {
            // Ball information (optimized)
            ball: {
                x: discs[0].x,
                y: discs[0].y,
                xs: discs[0].xspeed,
                ys: discs[0].yspeed,
                radius: discs[0].radius,
                invMass: discs[0].invMass,
                damping: discs[0].damping,
                // cMask and cGroup are IMPORTANT for collision prediction
                cMask: discs[0].cMask,
                cGroup: discs[0].cGroup
            },

            // Players information (optimized)
            players: playersArray.map(p => ({
                // Removed: name, avatar (not needed for AI)
                team: p.team.id,
                disc: p.disc ? {
                    x: p.disc.x,
                    y: p.disc.y,
                    xs: p.disc.xspeed,
                    ys: p.disc.yspeed,
                    radius: p.disc.radius,
                    invMass: p.disc.invMass,
                    damping: p.disc.damping,
                    // cMask and cGroup are CRITICAL for collision detection
                    cMask: p.disc.cMask,
                    cGroup: p.disc.cGroup
                } : null,
                inputs: p.inputs,
                bot: p.bot
            })),

            // Stadium info (optimized for AI training)
            stadium: {
                width: stadium.width,
                height: stadium.height,
                spawnDistance: stadium.spawnDistance,
                // Filter out kickOffBarrier segments and center barriers
                segments: stadium.segments ? stadium.segments.filter(s =>
                    s.trait !== 'kickOffBarrier' &&
                    !(s.v0 !== undefined && s.v1 !== undefined &&
                        Math.abs(stadium.vertexes[s.v0]?.x) < 50 &&
                        Math.abs(stadium.vertexes[s.v1]?.x) < 50)
                ) : [],
                goals: stadium.goals,
                // Filter out center circle and temporary discs
                discs: stadium.discs ? stadium.discs.filter(d =>
                    !(Math.abs(d.x) < 50 && Math.abs(d.y) < 50)
                ) : [],
                planes: stadium.planes,
                // Only keep essential traits
                traits: stadium.traits ? Object.fromEntries(
                    Object.entries(stadium.traits).filter(([key]) =>
                        key !== 'kickOffBarrier'
                    )
                ) : {},
                playerPhysics: stadium.playerPhysics,
                ballPhysics: stadium.ballPhysics
            }
        };
    },

    // Get bounding box of the playable field (from ballArea segments)
    getBoundingBox: getBoundingBox,

    // ─────────────────────────────────────────────────────────────────────────
    // getObs(agentTeam) → flat Float32Array, shape (101,)
    //
    // Section 1 — Field constants  [0..3]   4 features
    //   goal_y/NORM, HH/NORM, HW/NORM, agentTeam(0=RED,1=BLUE)
    // Section 2 — Agent ↔ Ball     [4..7]   4 features
    //   d_to_ball_x, d_to_ball_y, dist_to_ball, can_kick
    // Section 3 — Dynamic state    [8..16]  9 features
    //   ball_x, ball_y, ball_xs, ball_ys,
    //   my_x, my_y, my_xs, my_ys, my_speed
    // Section 4 — Game state       [17..18] 2 features
    //   time_remaining, possession
    // Section 5 — Teammates ×4    [19..54]  9×4=36 features  (pad 0 if absent)
    // Section 6 — Opponents  ×5   [55..99]  9×5=45 features  (pad 0 if absent)
    // Section 7 — Intercept       [100]     1 feature
    //   intercept_who: +1=agent, +0.5=teammate, -1=opponent, 0=none/unknown
    //
    // Per player (teammate/opp), 9 features:
    //   x/NORM, y/NORM, xs/MS, ys/MS,
    //   d_to_me_x/NORM, d_to_me_y/NORM,
    //   d_to_ball_x/NORM, d_to_ball_y/NORM,
    //   dist_to_ball/DIAG
    //
    // All x-coords are flipped for BLUE so both teams share one policy.
    // ─────────────────────────────────────────────────────────────────────────
    getObs: function (agentTeam) {
        // ── Constants ──
        const NORM = 800;
        const MAX_SPEED = 10;
        const DIAG = 1132;   // sqrt(800² + 800²)
        const PLAYER_R = 15;
        const BALL_R = 5.8;
        const KICK_DIST = 25;     // center-to-center
        const N_TM = 4;
        const N_OPP = 5;

        const flip = agentTeam === 2 ? -1 : 1;

        // ── Bounding box & goal ──
        const bbox = getBoundingBox();
        const HW = (bbox.W || NORM * 2) / 2;
        const HH = (bbox.H || NORM) / 2;

        let goal_y = 85; // default valn-v4
        if (stadium.goals && stadium.goals.length > 0) {
            const g = stadium.goals[0];
            goal_y = Math.abs(g.p0[1] - g.p1[1]) / 2;
        }

        // ── Ball ──
        const ball = discs[0];
        const bx = ball.x, by = ball.y;
        const bxs = ball.xspeed, bys = ball.yspeed;

        // ── Partition players ──
        let myPlayer = null;
        const teammates = [], opponents = [];
        for (const p of playersArray) {
            if (!p.disc) continue;
            if (p.team.id === agentTeam) {
                if (!myPlayer) myPlayer = p;
                else teammates.push(p);
            } else if (p.team.id !== 0) {
                opponents.push(p);
            }
        }

        const zero = { x: 0, y: 0, xspeed: 0, yspeed: 0 };
        const myD = myPlayer ? myPlayer.disc : zero;
        const mx = myD.x, my = myD.y, mxs = myD.xspeed, mys = myD.yspeed;

        // ── Helper: surface dist ball ──
        function distBall(px, py) {
            return Math.max(0, Math.hypot(px - bx, py - by) - PLAYER_R - BALL_R);
        }
        // ── Helper: surface dist to agent ──
        function distMe(px, py) {
            return Math.max(0, Math.hypot(px - mx, py - my) - 2 * PLAYER_R);
        }



        // ── Helper: encode one player (9 features) ──
        function encodePlayer(px, py, pxs, pys) {
            const db = distBall(px, py);
            return [
                flip * px / NORM,
                py / NORM,
                flip * pxs / MAX_SPEED,
                pys / MAX_SPEED,
                flip * (px - mx) / NORM,
                (py - my) / NORM,
                flip * (bx - px) / NORM,
                (by - py) / NORM,
                db / DIAG,
            ];
        }

        // Sort teammates & opponents by dist_to_ball ascending
        function distToBall(p) { return Math.hypot(p.disc.x - bx, p.disc.y - by); }
        teammates.sort((a, b) => distToBall(a) - distToBall(b));
        opponents.sort((a, b) => distToBall(a) - distToBall(b));

        // ── Build obs ──
        const obs = [];

        // Section 1 — Field (4)
        obs.push(goal_y / NORM);
        obs.push(HH / NORM);
        obs.push(HW / NORM);
        obs.push(agentTeam === 1 ? 0 : 1);

        // Section 2 — Agent ↔ Ball (4)
        const db_me = distBall(mx, my);
        obs.push(flip * (bx - mx) / NORM);           // d_to_ball_x
        obs.push((by - my) / NORM);                  // d_to_ball_y
        obs.push(db_me / DIAG);                      // dist_to_ball
        obs.push(Math.hypot(bx - mx, by - my) <= KICK_DIST ? 1 : 0); // can_kick

        // Section 3 — Dynamic state (9)
        obs.push(flip * bx / NORM);
        obs.push(by / NORM);
        obs.push(flip * bxs / MAX_SPEED);
        obs.push(bys / MAX_SPEED);
        obs.push(flip * mx / NORM);
        obs.push(my / NORM);
        obs.push(flip * mxs / MAX_SPEED);
        obs.push(mys / MAX_SPEED);
        obs.push(Math.hypot(mxs, mys) / MAX_SPEED);  // my_speed

        // Section 4 — Game state (2)
        // time_remaining: game object may expose timeLimit/time
        let timeRemaining = 0;
        if (typeof game !== 'undefined' && game.timeLimit > 0) {
            timeRemaining = Math.max(0, 1 - game.time / (game.timeLimit * 60));
        }
        obs.push(timeRemaining);

        // possession: sign(opp_closest_dist - my_closest_dist), mapped to [0,1]
        const myClosest = myPlayer ? Math.hypot(mx - bx, my - by) : Infinity;
        const oppClosest = opponents.length > 0
            ? Math.hypot(opponents[0].disc.x - bx, opponents[0].disc.y - by)
            : Infinity;
        const posVal = oppClosest < myClosest ? 1 : (myClosest < oppClosest ? -1 : 0);
        obs.push(posVal);  // -1, 0, or 1

        // Section 5 — Teammates ×N_TM (9 each = 36 total)
        for (let i = 0; i < N_TM; i++) {
            if (i < teammates.length) {
                const d = teammates[i].disc;
                obs.push(...encodePlayer(d.x, d.y, d.xspeed, d.yspeed));
            } else {
                for (let j = 0; j < 9; j++) obs.push(0);
            }
        }

        // Section 6 — Opponents ×N_OPP (9 each = 45 total)
        for (let i = 0; i < N_OPP; i++) {
            if (i < opponents.length) {
                const d = opponents[i].disc;
                obs.push(...encodePlayer(d.x, d.y, d.xspeed, d.yspeed));
            } else {
                for (let j = 0; j < 9; j++) obs.push(0);
            }
        }

        // obs.length should be 4+4+9+2+36+45 = 100

        return obs;
    },

    // ─────────────────────────────────────────────────────────────────────────
    // firstInterceptor(agentTeam, maxTicks)
    //
    // Simulates the ball's trajectory (up to maxTicks frames, WITH wall bounce)
    // and finds the FIRST player that could physically reach the ball's
    // position at that tick given their max cruising speed.
    //
    // Returns { player, teamId, isAgent, isTeammate, atTick } or null.
    //
    // Wall bounce: ball.bCoef × wall.bCoef (futsal walls = 1.0)
    // Stops if ball enters goal gap (|by| <= goal_y when crossing wall x).
    // maxSpeed ≈ acceleration / (1 - damping)  [Haxball physics steady-state]
    // ─────────────────────────────────────────────────────────────────────────
    firstInterceptor: function (agentTeam, maxTicks = 120, myPlayer = null) {
        const ball = discs[0];
        const ballSpeed = Math.hypot(ball.xspeed, ball.yspeed);
        // Skip expensive loop when ball is nearly stationary
        if (ballSpeed < 0.05) return null;

        // Determine player max speed from physics (use stadium override if present)
        const pPhys = (stadium && stadium.playerPhysics) ? stadium.playerPhysics : haxball.playerPhysics;
        const acc = pPhys.acceleration || haxball.playerPhysics.acceleration;
        const damp = pPhys.damping || haxball.playerPhysics.damping;
        // Steady-state max speed when holding one direction continuously
        const maxSpeed = acc / (1 - damp);

        // Pre-filter: only active field players
        const activePlayers = playersArray.filter(p => p.disc && p.team.id !== 0);
        if (activePlayers.length === 0) return null;

        // ── Field bounds & bounce coefficient ──
        const bbox = getBoundingBox();
        const HW = (bbox.W || 1400) / 2;
        const HH = (bbox.H || 640) / 2;
        let goal_y = 85;
        if (stadium.goals && stadium.goals.length > 0) {
            const g = stadium.goals[0];
            goal_y = Math.abs(g.p0[1] - g.p1[1]) / 2;
        }
        // Combined: ball.bCoef × wall.bCoef (futsal walls bCoef = 1.0)
        const wallBounce = (ball.bCoef !== undefined) ? ball.bCoef : 0.443;

        // Simulate ball position tick-by-tick WITH wall bounce
        let bx = ball.x, by = ball.y;
        let bxs = ball.xspeed, bys = ball.yspeed;
        const ballDamp = ball.damping;
        const ballRadius = ball.radius;

        for (let t = 1; t <= maxTicks; t++) {
            // 1. Move
            bx += bxs; by += bys;

            // 2. Wall bounce (axis-aligned, mirrors HaxBall resolveDPCollision)
            // ── Top / Bottom (always solid) ──
            if (by - ballRadius < -HH) {
                by = -HH + ballRadius;
                bys = -bys * wallBounce;
            } else if (by + ballRadius > HH) {
                by = HH - ballRadius;
                bys = -bys * wallBounce;
            }
            // ── Left wall ──
            if (bx - ballRadius < -HW) {
                if (Math.abs(by) <= goal_y) break;  // entered goal → stop
                bx = -HW + ballRadius;
                bxs = -bxs * wallBounce;
            }
            // ── Right wall ──
            else if (bx + ballRadius > HW) {
                if (Math.abs(by) <= goal_y) break;  // entered goal → stop
                bx = HW - ballRadius;
                bxs = -bxs * wallBounce;
            }

            // 3. Damping (after collision resolution)
            bxs *= ballDamp; bys *= ballDamp;

            // Check each player
            for (const p of activePlayers) {
                const pDisc = p.disc;
                const dist = Math.hypot(pDisc.x - bx, pDisc.y - by);
                // Touch distance: player surface to ball surface
                const touchDist = Math.max(0, dist - pDisc.radius - ballRadius);
                // Ticks needed at max speed (optimistic: ignores acceleration ramp-up)
                const ticksNeeded = touchDist / maxSpeed;
                if (ticksNeeded <= t) {
                    const isAgent = (myPlayer !== null) ? (p === myPlayer) : (p.team.id === agentTeam);
                    const isTeammate = !isAgent && (p.team.id === agentTeam);
                    return {
                        player: p,
                        teamId: p.team.id,
                        isAgent: isAgent,
                        isTeammate: isTeammate,
                        atTick: t,
                    };
                }
            }
        }
        return null; // no player can intercept within maxTicks
    },

    // ─────────────────────────────────────────────────────────────────────────
    // getInterceptReward(agentTeam, maxTicks)
    //
    // Convenience wrapper that converts firstInterceptor result into a scalar:
    //   +1   → agent (or teammate) intercepts first  (positive possession)
    //   -1   → opponent intercepts first
    //    0   → ball stationary or no one can reach in time
    // ─────────────────────────────────────────────────────────────────────────
    getInterceptReward: function (agentTeam, maxTicks = 120) {
        const result = this.firstInterceptor(agentTeam, maxTicks);
        if (!result) return 0;
        return result.teamId === agentTeam ? 1 : -1;
    },


    // Set input for a specific player (by index or name)
    setPlayerInput: function (playerIdentifier, input) {
        let player = null;

        if (typeof playerIdentifier === 'number') {
            player = playersArray[playerIdentifier];
        } else if (typeof playerIdentifier === 'string') {
            player = playersArray.find(p => p.name === playerIdentifier);
        }

        if (!player) {
            console.error('Player not found:', playerIdentifier);
            return false;
        }

        // Set inputs using Input constants
        let inputs = 0;
        if (input.up) inputs += Input.UP;
        if (input.down) inputs += Input.DOWN;
        if (input.left) inputs += Input.LEFT;
        if (input.right) inputs += Input.RIGHT;
        if (input.kick) inputs += Input.SHOOT;

        player.inputs = inputs;
        return true;
    },

    // Utility: Get distance between two points
    distance: function (p1, p2) {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    },

    // Utility: Get angle from p1 to p2
    angleTo: function (p1, p2) {
        return Math.atan2(p2.y - p1.y, p2.x - p1.x);
    },

    // Utility: Predict ball position after N ticks (simplified)
    predictBallPosition: function (ticks) {
        const ball = discs[0];
        const damping = ball.damping;

        let x = ball.x;
        let y = ball.y;
        let xspeed = ball.xspeed;
        let yspeed = ball.yspeed;

        for (let i = 0; i < ticks; i++) {
            x += xspeed;
            y += yspeed;
            xspeed *= damping;
            yspeed *= damping;

            // Simple wall bounce (not perfect but close enough)
            if (Math.abs(x) > stadium.width) {
                xspeed = -xspeed * ball.bCoef;
                x = Math.sign(x) * stadium.width;
            }
            if (Math.abs(y) > stadium.height) {
                yspeed = -yspeed * ball.bCoef;
                y = Math.sign(y) * stadium.height;
            }
        }

        return { x, y };
    },

    // Add a new bot player
    addBot: function (name, team, botFunction) {
        const teamObj = team === 1 ? haxball.Team.RED :
            team === 2 ? haxball.Team.BLUE :
                haxball.Team.SPECTATORS;

        const newPlayer = new Player();
        newPlayer.init(name, '🤖', teamObj, [], botFunction);
        setPlayerDefaultProperties(newPlayer);
        playersArray.push(newPlayer);

        return playersArray.length - 1; // Return player index
    },

    // Input constants (expose for convenience)
    Input: Input
};

console.log('✅ Agent API loaded! Use AgentAPI.getState() to read game state.');

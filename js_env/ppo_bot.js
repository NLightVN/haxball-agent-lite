/**
 * ppo_bot.js — PPOBot: runs a1_final.onnx inside the browser via onnxruntime-web.
 *
 * API (called by ppoBot() in bot.js):
 *   PPOBot.loaded          — true once ONNX model is ready
 *   PPOBot.update(player)  — call each game frame; sets player.inputs
 *   PPOBot.load(url)       — (auto-called) load ONNX model
 *
 * How to add the bot from the browser console:
 *   AgentAPI.addBot("A1", 1, ppoBot)   // team 1 = RED
 *   AgentAPI.addBot("A1", 2, ppoBot)   // team 2 = BLUE
 */

window.PPOBot = (() => {
    // ── Direction map (matches Python DIR_MAP) ─────────────────────────────
    // idx → [Input flags]
    // UP: 1, LEFT: 2, DOWN: 4, RIGHT: 8, SHOOT: 16
    const DIR_INPUT = [
        0,         // 0: stay
        8,         // 1: right
        2,         // 2: left
        1,         // 3: up
        4,         // 4: down
        8 | 1,     // 5: up-right
        2 | 1,     // 6: up-left
        8 | 4,     // 7: down-right
        2 | 4,     // 8: down-left
    ];

    const FRAME_SKIP = 6;   // decision every 6 physics ticks (matches training)
    const OBS_DIM = 106;
    const N_DIR = 9;
    const N_KICK = 2;

    // ── State ──────────────────────────────────────────────────────────────
    let session = null;
    let loaded = false;
    let loading = false;

    // per-player state (keyed by player object reference)
    const playerState = new WeakMap();

    // ── Helpers ────────────────────────────────────────────────────────────
    function argmax(arr, start, len) {
        let best = start, bestVal = arr[start];
        for (let i = start + 1; i < start + len; i++) {
            if (arr[i] > bestVal) { bestVal = arr[i]; best = i; }
        }
        return best - start;
    }

    // ── Load ONNX model ────────────────────────────────────────────────────
    async function load(url = 'models/a1_final.onnx') {
        if (loading || loaded) return;
        loading = true;
        try {
            if (typeof ort === 'undefined') {
                console.error('[PPOBot] onnxruntime-web (ort) is not loaded. Add the CDN script before ppo_bot.js.');
                return;
            }
            console.log(`[PPOBot] Loading ONNX model from ${url} ...`);
            session = await ort.InferenceSession.create(url, {
                executionProviders: ['wasm'],
            });
            loaded = true;
            console.log('✅ PPOBot loaded — add bot: AgentAPI.addBot("A1", 1, ppoBot)');
        } catch (e) {
            console.error('[PPOBot] Failed to load model:', e);
        } finally {
            loading = false;
        }
    }

    // ── Inference ──────────────────────────────────────────────────────────
    async function runInference(obsArray) {
        if (!session) return null;
        try {
            const tensor = new ort.Tensor('float32', new Float32Array(obsArray), [1, OBS_DIM]);
            const results = await session.run({ obs: tensor });
            return results.logits.data;   // Float32Array of length 11
        } catch (e) {
            console.warn('[PPOBot] Inference error:', e);
            return null;
        }
    }

    // ── update(player) — called every frame by ppoBot() in bot.js ──────────
    function update(player) {
        if (!loaded) return;

        // Init per-player state
        if (!playerState.has(player)) {
            playerState.set(player, { frameCount: 0, pendingInput: 0, inferring: false });
        }
        const state = playerState.get(player);
        state.frameCount++;

        // Apply last committed action every frame (smooth movement)
        player.inputs = state.pendingInput;

        // Kick off new inference every FRAME_SKIP frames (async, non-blocking)
        if (state.frameCount % FRAME_SKIP === 0 && !state.inferring) {
            const agentTeam = player.team.id;
            const obs = AgentAPI.getObs(agentTeam);

            if (obs && obs.length === OBS_DIM) {
                state.inferring = true;
                runInference(obs).then(logits => {
                    state.inferring = false;
                    if (!logits) return;

                    // Decode dir: first N_DIR logits
                    const dirIdx = argmax(logits, 0, N_DIR);
                    // Decode kick: next N_KICK logits
                    const kickIdx = argmax(logits, N_DIR, N_KICK);

                    let input = DIR_INPUT[dirIdx] || 0;
                    
                    // Flip left/right actions if agent is on BLUE team
                    if (agentTeam === 2) {
                        let flippedInput = input & ~10; // 10 is 8 (RIGHT) | 2 (LEFT)
                        if (input & 8) flippedInput |= 2; // RIGHT -> LEFT
                        if (input & 2) flippedInput |= 8; // LEFT -> RIGHT
                        input = flippedInput;
                    }

                    if (kickIdx === 1) {
                        let inRange = true;
                        // Hidden buff: only kick if ball is actually in kick range (4.0)
                        if (player.disc && typeof discs !== 'undefined' && discs[0]) {
                            const ball = discs[0];
                            const pDisc = player.disc;
                            const dx = ball.x - pDisc.x;
                            const dy = ball.y - pDisc.y;
                            const dist = Math.sqrt(dx*dx + dy*dy);
                            if (dist - ball.radius - pDisc.radius > 4.0) {
                                inRange = false;
                            }
                        }
                        if (inRange) input |= 16;
                    }

                    state.pendingInput = input;
                });
            }
        }
    }

    // Auto-load on script start
    load();

    return { get loaded() { return loaded; }, update, load };
})();

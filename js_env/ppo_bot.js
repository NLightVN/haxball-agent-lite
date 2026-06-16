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
    const DIR_INPUT = [
        0,                                    // 0: stay
        Input.RIGHT,                          // 1: right
        Input.LEFT,                           // 2: left
        Input.UP,                             // 3: up
        Input.DOWN,                           // 4: down
        Input.RIGHT | Input.UP,               // 5: up-right
        Input.LEFT | Input.UP,               // 6: up-left
        Input.RIGHT | Input.DOWN,             // 7: down-right
        Input.LEFT | Input.DOWN,             // 8: down-left
    ];

    const FRAME_SKIP = 3;   // decision every 3 physics ticks (matches training)
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
                    if (kickIdx === 1) input |= Input.SHOOT;

                    state.pendingInput = input;
                });
            }
        }
    }

    // Auto-load on script start
    load();

    return { get loaded() { return loaded; }, update, load };
})();

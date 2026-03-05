/**
 * ppo_bot_b0.js — B0 Defend Bot: runs b0_best.onnx inside the browser.
 *
 * B0 defends its own goal. Use as OPPONENT (solo mode = you attack, B0 defends).
 *
 * How to add from browser console (after page loads):
 *   // B0 defends LEFT goal (RED side) — you play BLUE
 *   AgentAPI.addBot("B0", 1, ppoBotB0)
 *
 *   // B0 defends RIGHT goal (BLUE side) — you play RED
 *   AgentAPI.addBot("B0", 2, ppoBotB0)
 *
 * Controls: SPACE = pause | R = restart (use game controls)
 */

window.PPOBotB0 = (() => {
    const DIR_INPUT = [
        0,                              // 0: stay
        Input.RIGHT,                    // 1: right
        Input.LEFT,                     // 2: left
        Input.UP,                       // 3: up
        Input.DOWN,                     // 4: down
        Input.RIGHT | Input.UP,         // 5: up-right
        Input.LEFT  | Input.UP,         // 6: up-left
        Input.RIGHT | Input.DOWN,       // 7: down-right
        Input.LEFT  | Input.DOWN,       // 8: down-left
    ];

    const FRAME_SKIP = 6;
    const OBS_DIM    = 106;
    const N_DIR      = 9;
    const N_KICK     = 2;

    let session  = null;
    let loaded   = false;
    let loading  = false;

    const playerState = new WeakMap();

    function argmax(arr, start, len) {
        let best = start, bestVal = arr[start];
        for (let i = start + 1; i < start + len; i++) {
            if (arr[i] > bestVal) { bestVal = arr[i]; best = i; }
        }
        return best - start;
    }

    async function load(url = 'models/b0_best.onnx') {
        if (loading || loaded) return;
        loading = true;
        try {
            if (typeof ort === 'undefined') {
                console.error('[PPOBotB0] onnxruntime-web (ort) not loaded.');
                return;
            }
            console.log(`[PPOBotB0] Loading ONNX model from ${url} ...`);
            session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
            loaded = true;
            console.log('✅ PPOBotB0 loaded — solo mode: AgentAPI.addBot("B0", 1, ppoBotB0)');
        } catch (e) {
            console.error('[PPOBotB0] Failed to load:', e);
        } finally {
            loading = false;
        }
    }

    async function runInference(obsArray) {
        if (!session) return null;
        try {
            const tensor  = new ort.Tensor('float32', new Float32Array(obsArray), [1, OBS_DIM]);
            const results = await session.run({ obs: tensor });
            return results.logits.data;
        } catch (e) {
            console.warn('[PPOBotB0] Inference error:', e);
            return null;
        }
    }

    function update(player) {
        if (!loaded) return;

        if (!playerState.has(player)) {
            playerState.set(player, { frameCount: 0, pendingInput: 0, inferring: false });
        }
        const state = playerState.get(player);
        state.frameCount++;
        player.inputs = state.pendingInput;

        if (state.frameCount % FRAME_SKIP === 0 && !state.inferring) {
            const agentTeam = player.team.id;
            const obs = AgentAPI.getObs(agentTeam);

            if (obs && obs.length === OBS_DIM) {
                state.inferring = true;
                runInference(obs).then(logits => {
                    state.inferring = false;
                    if (!logits) return;
                    const dirIdx  = argmax(logits, 0,     N_DIR);
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

// Convenience alias used by AgentAPI.addBot
window.ppoBotB0 = player => PPOBotB0.update(player);

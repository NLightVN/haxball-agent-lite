// copilot-bot.js
// Haxball bot powered by GitHub Copilot / GitHub Models API.
// Tracks every API call so you can see exactly how many "lượt Copilot" were used.
//
// Quick start (DevTools console):
//   const bot = new CopilotBot(1, 1, '<your-github-token>').start();
//   CopilotUsageTracker.count   // → current usage count

// ── Global Usage Tracker ──────────────────────────────────────────────────────
// A single shared counter for all CopilotBot instances on the page.
window.CopilotUsageTracker = {
    _count: 0,

    increment() {
        this._count++;
        this._updateDisplay();
    },

    reset() {
        this._count = 0;
        this._updateDisplay();
    },

    get count() { return this._count; },

    _updateDisplay() {
        const el = document.getElementById('copilot-usage-count');
        if (!el) return;
        el.textContent = this._count;
        // Remove and re-add the 'bump' class to restart the CSS scale animation.
        // The void/offsetWidth forces a reflow so the browser registers the class removal
        // before re-adding it — without this the animation won't replay.
        el.classList.remove('bump');
        void el.offsetWidth;
        el.classList.add('bump');
        setTimeout(() => el.classList.remove('bump'), 200);
    }
};

// ── CopilotBot ────────────────────────────────────────────────────────────────
class CopilotBot {
    // GitHub Models endpoint (free tier, OpenAI-compatible, uses a GitHub token).
    // You can also point this at https://api.githubcopilot.com/chat/completions
    // for a Copilot-subscription endpoint.
    static DEFAULT_ENDPOINT = 'https://models.inference.ai.azure.com/chat/completions';
    static DEFAULT_MODEL    = 'gpt-4o-mini';

    /**
     * @param {number} playerIndex  Index in playersArray (0 = first player, 1 = second, …)
     * @param {number} team         1 = RED, 2 = BLUE
     * @param {string} githubToken  Personal Access Token with "models:read" scope
     * @param {object} [options]
     * @param {string} [options.endpoint]    Override API endpoint
     * @param {string} [options.model]       Override model name
     * @param {number} [options.intervalMs]  Milliseconds between API calls (default 400)
     */
    constructor(playerIndex, team, githubToken, options = {}) {
        this.playerIndex = playerIndex;
        this.team        = team;
        this.token       = githubToken;
        this.endpoint    = options.endpoint    || CopilotBot.DEFAULT_ENDPOINT;
        this.model       = options.model       || CopilotBot.DEFAULT_MODEL;
        this.intervalMs  = options.intervalMs  || 400;
        this.temperature = options.temperature !== undefined ? options.temperature : 0.0;

        this._lastInput    = { up: false, down: false, left: false, right: false, kick: false };
        this._thinking     = false;
        this._intervalId   = null;
        this._sessionUsage = 0;
    }

    // ── Internal: build prompt from live game state ───────────────────────────
    _buildPrompt() {
        if (!window.AgentAPI) return null;

        const state = AgentAPI.getState();
        const me    = state.players[this.playerIndex];
        if (!me || !me.disc) return null;

        const ball   = state.ball;
        const myDisc = me.disc;
        const goal   = this.team === 1 ? 'right side' : 'left side';

        return [
            `You are controlling a Haxball player.`,
            `Your team: ${this.team === 1 ? 'RED' : 'BLUE'} — score on the ${goal}.`,
            `Ball  : x=${ball.x.toFixed(1)}, y=${ball.y.toFixed(1)}, vx=${ball.xs.toFixed(2)}, vy=${ball.ys.toFixed(2)}`,
            `Player: x=${myDisc.x.toFixed(1)}, y=${myDisc.y.toFixed(1)}, vx=${myDisc.xs.toFixed(2)}, vy=${myDisc.ys.toFixed(2)}`,
            ``,
            `Reply with ONLY a JSON object and no other text:`,
            `{"up":bool,"down":bool,"left":bool,"right":bool,"kick":bool}`
        ].join('\n');
    }

    // ── Internal: call the Copilot/Models API ─────────────────────────────────
    async _askCopilot() {
        const prompt = this._buildPrompt();
        if (!prompt) return null;

        const response = await fetch(this.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify({
                model: this.model,
                messages: [{ role: 'user', content: prompt }],
                max_tokens: 40,
                temperature: this.temperature
            })
        });

        if (!response.ok) {
            console.warn(`[CopilotBot] API error ${response.status}: ${response.statusText}`);
            return null;
        }

        const data    = await response.json();
        const content = (data.choices?.[0]?.message?.content || '').trim();
        // Extract the JSON object from the response. The prompt asks for a flat
        // object with only boolean values, so we scan for the first complete {...}
        // block by tracking brace depth rather than using a simple character class.
        let jsonStr = null;
        const start = content.indexOf('{');
        if (start !== -1) {
            let depth = 0, end = -1;
            for (let i = start; i < content.length; i++) {
                if (content[i] === '{') depth++;
                else if (content[i] === '}') { depth--; if (depth === 0) { end = i; break; } }
            }
            if (end !== -1) jsonStr = content.slice(start, end + 1);
        }
        if (!jsonStr) {
            console.warn('[CopilotBot] Could not extract JSON from response:', content);
            return null;
        }

        return JSON.parse(jsonStr);
    }

    // ── Internal: one think-cycle ─────────────────────────────────────────────
    async _think() {
        if (this._thinking) return;
        this._thinking = true;

        try {
            const input = await this._askCopilot();
            if (input) {
                this._lastInput = input;
                this._sessionUsage++;
                CopilotUsageTracker.increment();
            }
        } catch (err) {
            console.warn('[CopilotBot] Request failed:', err.message);
        } finally {
            this._thinking = false;
        }

        // Always apply last known input (keeps player moving even between API calls)
        if (window.AgentAPI) {
            AgentAPI.setPlayerInput(this.playerIndex, this._lastInput);
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /** Start the bot loop. Returns `this` for chaining. */
    start() {
        if (this._intervalId) this.stop();
        this._intervalId = setInterval(() => this._think(), this.intervalMs);
        console.log(`[CopilotBot] Started — player ${this.playerIndex}, team ${this.team}, interval ${this.intervalMs}ms`);
        return this;
    }

    /** Stop the bot loop. Returns `this` for chaining. */
    stop() {
        if (this._intervalId) {
            clearInterval(this._intervalId);
            this._intervalId = null;
        }
        if (window.AgentAPI) {
            AgentAPI.setPlayerInput(this.playerIndex,
                { up: false, down: false, left: false, right: false, kick: false });
        }
        console.log(`[CopilotBot] Stopped — session usage: ${this._sessionUsage} lượt`);
        return this;
    }

    /** Number of successful API calls made by this bot instance this session. */
    get sessionUsage() { return this._sessionUsage; }
}

// ── Settings Panel (token management) ────────────────────────────────────────
(function initCopilotSettings() {
    const STORAGE_KEY = 'copilot_github_token';

    window.CopilotSettings = {
        getToken()  { return localStorage.getItem(STORAGE_KEY) || ''; },
        saveToken(t){ localStorage.setItem(STORAGE_KEY, t); },

        /** Basic sanity check: GitHub tokens start with ghp_, gho_, ghu_, ghs_, or github_pat_ */
        isValidToken(t) {
            return /^(ghp_|gho_|ghu_|ghs_|github_pat_)\S{10,}/.test(t);
        },

        openPanel() {
            const panel = document.getElementById('copilot-settings-panel');
            if (panel) panel.style.display = 'flex';
            const inp = document.getElementById('copilot-token-input');
            if (inp) inp.value = this.getToken();
        },

        closePanel() {
            const panel = document.getElementById('copilot-settings-panel');
            if (panel) panel.style.display = 'none';
        },

        saveFromPanel() {
            const inp = document.getElementById('copilot-token-input');
            if (!inp) return;
            const token = inp.value.trim();
            if (token && !this.isValidToken(token)) {
                console.warn('[CopilotBot] Token format looks incorrect. Expected: ghp_…, github_pat_…, etc.');
            }
            this.saveToken(token);
            this.closePanel();
            console.log('[CopilotBot] GitHub token saved to localStorage.');
        }
    };
})();

console.log('✅ Copilot Bot loaded!');
console.log('   Usage: new CopilotBot(playerIndex, team, token).start()');
console.log('   Usage count: CopilotUsageTracker.count');

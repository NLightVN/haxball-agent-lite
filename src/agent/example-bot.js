// Example AI Bot - Simple chasing behavior
// This demonstrates how to use the GameAPI to create an AI agent

class ExampleBot {
    constructor() {
        this.updateInterval = null;
        this.tickRate = 60; // Update 60 times per second (same as Haxball)
    }

    // Start the bot
    start() {
        if (this.updateInterval) return;

        console.log('🤖 AI Bot started');
        this.updateInterval = setInterval(() => {
            this.update();
        }, 1000 / this.tickRate);
    }

    // Stop the bot
    stop() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            console.log('🤖 AI Bot stopped');
        }
    }

    // Main update loop - called every tick
    update() {
        const state = GameAPI.getState();
        if (!state) return;

        // Find our player (Blue team - Player 2)
        const me = state.players.find(p => p.team === 2);
        if (!me || !me.disc) return;

        const ball = state.ball;
        if (!ball) return;

        // Simple AI strategy: Chase the ball and try to kick it towards opponent goal
        const input = this.calculateInput(me, ball, state);
        GameAPI.setAgentInput(input);
    }

    // Calculate what input to send based on current state
    calculateInput(me, ball, state) {
        const myPos = me.disc;

        // Target: try to position between ball and own goal, then push ball forward
        const targetX = ball.x - 20; // Position slightly behind the ball
        const targetY = ball.y;

        // Calculate direction to target
        const dx = targetX - myPos.x;
        const dy = targetY - myPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Movement input
        const input = {
            up: false,
            down: false,
            left: false,
            right: false,
            kick: false
        };

        // Move towards target
        const moveThreshold = 5; // Dead zone
        if (Math.abs(dx) > moveThreshold) {
            input.left = dx < 0;
            input.right = dx > 0;
        }
        if (Math.abs(dy) > moveThreshold) {
            input.up = dy < 0;
            input.down = dy > 0;
        }

        // Kick if close to ball and ball is moving away from our goal
        const distanceToBall = GameAPI.distance(myPos, ball);
        const kickDistance = 25; // Kick range

        if (distanceToBall < kickDistance) {
            // Kick if ball is in front of us (towards opponent goal)
            if (ball.x > myPos.x) {
                input.kick = true;
            }
        }

        return input;
    }
}

// Create bot instance
window.exampleBot = new ExampleBot();

// Simple toggle function
window.toggleAgent = function () {
    const btn = document.getElementById('agent-btn');
    const enabled = GameAPI.agentEnabled;

    if (enabled) {
        // Disable agent
        GameAPI.setAgentEnabled(false);
        window.exampleBot.stop();
        btn.classList.remove('active');
        btn.textContent = '🤖 Enable AI Agent (Player 2)';
    } else {
        // Enable agent
        GameAPI.setAgentEnabled(true);
        window.exampleBot.start();
        btn.classList.add('active');
        btn.textContent = '🤖 Disable AI Agent (Player 2)';
    }
};

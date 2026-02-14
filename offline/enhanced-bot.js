// Enhanced Example Bot using AgentAPI
// Demonstrates how to create custom AI bots

class EnhancedBot {
    constructor(playerIndex) {
        this.playerIndex = playerIndex;
        this.name = `Bot ${playerIndex}`;
    }

    // This is called every frame during game loop
    update() {
        const state = AgentAPI.getState();
        const me = state.players[this.playerIndex];

        if (!me || !me.disc) return;

        const ball = state.ball;
        const myDisc = me.disc;

        // Calculate ball distance
        const distToBall = AgentAPI.distance(myDisc, ball);

        // Simple strategy: chase ball and try to kick towards opponent goal
        const input = {
            up: false,
            down: false,
            left: false,
            right: false,
            kick: false
        };

        // Movement threshold
        const threshold = 3;

        // Move towards ball
        if (ball.x - myDisc.x > threshold) input.right = true;
        if (ball.x - myDisc.x < -threshold) input.left = true;
        if (ball.y - myDisc.y > threshold) input.down = true;
        if (ball.y - myDisc.y < -threshold) input.up = true;

        // Kick when close to ball
        const kickRange = 25;
        if (distToBall < kickRange) {
            // Only kick if ball is in front (towards opponent goal)
            if (me.team === 1 && ball.x > myDisc.x) {  // Red team kicks right
                input.kick = true;
            } else if (me.team === 2 && ball.x < myDisc.x) {  // Blue team kicks left
                input.kick = true;
            }
        }

        // Set the inputs
        AgentAPI.setPlayerInput(this.playerIndex, input);
    }
}

// Example: How to use the bot
// Uncomment these lines to add a bot to the game:

// window.myBot = new EnhancedBot(1); // Control player at index 1
// 
// // Then in the game loop (or enable via button), call:
// // myBot.update();

console.log('✅ Enhanced Bot class loaded! Create with: new EnhancedBot(playerIndex)');

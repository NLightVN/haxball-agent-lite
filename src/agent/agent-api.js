// Agent API - Interface for AI bots to interact with the game
// Exposes game state and allows controlling Player 2

window.GameAPI = {
    room: null,
    agentEnabled: false,

    // Initialize the API with room reference
    init: function (roomRef) {
        this.room = roomRef;
    },

    // Get current game state
    getState: function () {
        if (!this.room) return null;

        try {
            const players = this.room.getPlayerList();
            const scores = this.room.getScores();
            const ballPos = this.room.getBallPosition();

            // Get detailed disc properties
            const ball = this.room.getDiscProperties(0);

            const state = {
                // Ball state
                ball: ball ? {
                    x: ball.x,
                    y: ball.y,
                    xspeed: ball.xspeed || 0,
                    yspeed: ball.yspeed || 0,
                    radius: ball.radius
                } : ballPos,

                // Players state
                players: players.map(p => {
                    const discProps = this.room.getPlayerDiscProperties(p.id);
                    return {
                        id: p.id,
                        name: p.name,
                        team: p.team,
                        position: p.position,
                        disc: discProps ? {
                            x: discProps.x,
                            y: discProps.y,
                            xspeed: discProps.xspeed || 0,
                            yspeed: discProps.yspeed || 0,
                            radius: discProps.radius
                        } : p.position
                    };
                }),

                // Game state
                score: {
                    red: scores ? scores.red : 0,
                    blue: scores ? scores.blue : 0,
                    time: scores ? scores.time : 0,
                    timeLimit: scores ? scores.timeLimit : 180
                },

                // Stadium dimensions (Classic map)
                stadium: {
                    width: 420,
                    height: 200
                }
            };

            return state;
        } catch (e) {
            console.error('Error getting game state:', e);
            return null;
        }
    },

    // Set agent input (controls Player 2)
    setAgentInput: function (input) {
        if (!this.room || !this.agentEnabled) return;

        try {
            const players = this.room.getPlayerList();
            const player2 = players.find(p => p.team === 2);

            if (!player2) return;

            // Haxball Headless API doesn't have direct input control
            // We need to use a workaround by sending input events
            // This will be handled by the game.js keyboard override
            window.agentInput = {
                up: input.up || false,
                down: input.down || false,
                left: input.left || false,
                right: input.right || false,
                kick: input.kick || false
            };
        } catch (e) {
            console.error('Error setting agent input:', e);
        }
    },

    // Enable/disable agent control
    setAgentEnabled: function (enabled) {
        this.agentEnabled = enabled;
        if (!enabled) {
            window.agentInput = null;
        }
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

    // Utility: Predict ball position after N ticks
    predictBallPosition: function (ticks) {
        const state = this.getState();
        if (!state || !state.ball) return null;

        const ball = state.ball;
        const damping = 0.99;

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
            if (Math.abs(x) > 370) {
                xspeed = -xspeed * 0.5;
                x = Math.sign(x) * 370;
            }
            if (Math.abs(y) > 170) {
                yspeed = -yspeed * 0.5;
                y = Math.sign(y) * 170;
            }
        }

        return { x, y };
    }
};

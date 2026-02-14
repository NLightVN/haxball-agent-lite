// Agent API - Wrapper to access game state and control bots
// Built on top of Wazarr94's Haxball clone physics engine

window.AgentAPI = {
    // Get current game state
    getState: function () {
        return {
            // Ball state (disc[0])
            ball: {
                x: discs[0].x,
                y: discs[0].y,
                xspeed: discs[0].xspeed,
                yspeed: discs[0].yspeed,
                radius: discs[0].radius
            },

            // All players
            players: playersArray.map(p => ({
                name: p.name,
                team: p.team.id, // 0=spec, 1=red, 2=blue
                avatar: p.avatar,
                disc: p.disc ? {
                    x: p.disc.x,
                    y: p.disc.y,
                    xspeed: p.disc.xspeed,
                    yspeed: p.disc.yspeed,
                    radius: p.disc.radius
                } : null,
                inputs: p.inputs,
                bot: p.bot
            })),

            // Game state
            score: {
                red: game.red,
                blue: game.blue,
                time: game.time,
                timeLimit: game.timeLimit,
                scoreLimit: game.scoreLimit
            },

            // Stadium info
            stadium: {
                name: stadium.name,
                width: stadium.width,
                height: stadium.height
            },

            // Current frame
            frame: currentFrame
        };
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

// Main Game Script - Initializes Haxball Headless room and handles controls

(function () {
    'use strict';

    let room = null;
    let gameInitialized = false;
    const iframe = document.getElementById('game-frame');

    // Player input states
    const player1Input = {
        up: false,
        down: false,
        left: false,
        right: false,
        kick: false
    };

    const player2Input = {
        up: false,
        down: false,
        left: false,
        right: false,
        kick: false
    };

    // Initialize game when iframe loads
    iframe.addEventListener('load', function () {
        console.log('Iframe loaded, waiting for Haxball API...');
        updateStatus('⏳ Waiting for Haxball Headless API...', 'loading');

        // Wait a bit for Haxball to be ready, then inject our script
        setTimeout(initializeGame, 2000);
    });

    function initializeGame() {
        try {
            console.log('Attempting to initialize game...');

            // Inject game initialization script into iframe
            const initScript = `
                (function() {
                    // Check if HBInit is available
                    if (typeof HBInit === 'undefined') {
                        console.error('HBInit not found!');
                        window.parent.postMessage({type: 'error', message: 'HBInit not available'}, '*');
                        return;
                    }
                    
                    console.log('HBInit found, creating room...');
                    
                    // Room configuration
                    const roomConfig = {
                        roomName: "Haxball Agent Lite - 2P Local",
                        playerName: "Player 1",
                        maxPlayers: 2,
                        public: false,
                        noPlayer: false,
                        token: "" // Can add token if needed
                    };
                    
                    // Initialize room
                    try {
                        const room = HBInit(roomConfig);
                        console.log('Room created successfully!');
                        
                        // Set classic stadium
                        room.setDefaultStadium("Classic");
                        room.setScoreLimit(3);
                        room.setTimeLimit(3);
                        
                        // Start game immediately when 2 players join
                        room.onPlayerJoin = function(player) {
                            console.log(player.name + " joined");
                            
                            // Auto-assign teams
                            if (player.id === 1) {
                                room.setPlayerTeam(player.id, 1); // Red
                            } else if (player.id === 2) {
                                room.setPlayerTeam(player.id, 2); // Blue
                            }
                            
                            // Start game when we have 2 players
                            const players = room.getPlayerList();
                            if (players.length === 2) {
                                room.startGame();
                                console.log('Game started with 2 players!');
                            }
                            
                            // Notify parent
                            window.parent.postMessage({
                                type: 'playerJoin',
                                player: {id: player.id, name: player.name, team: player.team}
                            }, '*');
                        };
                        
                        room.onPlayerLeave = function(player) {
                            console.log(player.name + " left");
                        };
                        
                        room.onTeamGoal = function(team) {
                            const scores = room.getScores();
                            console.log('Goal scored by team ' + team);
                            
                            // Notify parent to update score display
                            window.parent.postMessage({
                                type: 'goal',
                                scores: scores
                            }, '*');
                        };
                        
                        room.onGameStart = function() {
                            console.log('Game started!');
                        };
                        
                        room.onGameStop = function() {
                            console.log('Game stopped');
                        };
                        
                        room.onGameTick = function() {
                            // Send game state to parent every tick
                            const scores = room.getScores();
                            const ball = room.getBallPosition();
                            
                            window.parent.postMessage({
                                type: 'tick',
                                scores: scores,
                                ball: ball
                            }, '*');
                        };
                        
                        // Expose room to parent window
                        window.haxRoom = room;
                        
                        // Notify parent that room is ready
                        window.parent.postMessage({
                            type: 'roomReady',
                            roomLink: room ? 'Room created (private)' : 'No link'
                        }, '*');
                        
                    } catch (error) {
                        console.error('Error creating room:', error);
                        window.parent.postMessage({
                            type: 'error',
                            message: 'Failed to create room: ' + error.message
                        }, '*');
                    }
                })();
            `;

            // Inject script into iframe
            iframe.contentWindow.postMessage({
                type: 'eval',
                code: initScript
            }, '*');

            // Alternative: use eval if message doesn't work
            setTimeout(() => {
                try {
                    iframe.contentWindow.eval(initScript);
                } catch (e) {
                    console.log('Direct eval not allowed, using postMessage');
                }
            }, 500);

        } catch (error) {
            console.error('Error initializing game:', error);
            updateStatus('❌ Error: ' + error.message, 'error');
        }
    }

    // Listen for messages from iframe
    window.addEventListener('message', function (event) {
        const data = event.data;

        if (data.type === 'roomReady') {
            console.log('Room ready!', data.roomLink);
            gameInitialized = true;

            // Get room reference from iframe
            try {
                room = iframe.contentWindow.haxRoom;
                GameAPI.init(room);
                updateStatus('✅ Room created! Waiting for Player 2...', 'ready');
                document.getElementById('room-status').textContent = '🟢 Online';
            } catch (e) {
                console.error('Could not access room from iframe:', e);
            }
        }

        else if (data.type === 'playerJoin') {
            console.log('Player joined:', data.player);
            if (data.player.id === 2) {
                updateStatus('✅ Game ready! Both players joined.', 'ready');
            }
        }

        else if (data.type === 'goal') {
            updateScoreDisplay(data.scores);
        }

        else if (data.type === 'tick') {
            updateGameDisplay(data.scores, data.ball);
        }

        else if (data.type === 'error') {
            console.error('Error from iframe:', data.message);
            updateStatus('❌ ' + data.message, 'error');
        }
    });

    // Keyboard controls for Player 1 and Player 2
    document.addEventListener('keydown', function (e) {
        handleKeyInput(e, true);
    });

    document.addEventListener('keyup', function (e) {
        handleKeyInput(e, false);
    });

    function handleKeyInput(e, isDown) {
        // Prevent default for game keys
        const gameKeys = ['w', 'a', 's', 'd', ' ', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Enter'];
        if (gameKeys.includes(e.key)) {
            e.preventDefault();
        }

        // Player 1 (WASD + Space)
        if (e.key === 'w' || e.key === 'W') player1Input.up = isDown;
        if (e.key === 's' || e.key === 'S') player1Input.down = isDown;
        if (e.key === 'a' || e.key === 'A') player1Input.left = isDown;
        if (e.key === 'd' || e.key === 'D') player1Input.right = isDown;
        if (e.key === ' ') player1Input.kick = isDown;

        // Player 2 (Arrow keys + Enter) - only if agent is not enabled
        if (!GameAPI.agentEnabled) {
            if (e.key === 'ArrowUp') player2Input.up = isDown;
            if (e.key === 'ArrowDown') player2Input.down = isDown;
            if (e.key === 'ArrowLeft') player2Input.left = isDown;
            if (e.key === 'ArrowRight') player2Input.right = isDown;
            if (e.key === 'Enter') player2Input.kick = isDown;
        }

        // Send input to iframe (this would need Haxball API support for direct input)
        // For now, players control via the iframe directly
    }

    function updateScoreDisplay(scores) {
        if (!scores) return;
        document.getElementById('score').textContent = `${scores.red} - ${scores.blue}`;
    }

    function updateGameDisplay(scores, ball) {
        if (scores) {
            updateScoreDisplay(scores);

            // Update time
            const timeLeft = scores.timeLimit - scores.time;
            const minutes = Math.floor(timeLeft / 60);
            const seconds = Math.floor(timeLeft % 60);
            document.getElementById('time').textContent =
                `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    function updateStatus(message, className) {
        const statusEl = document.getElementById('status');
        statusEl.textContent = message;
        statusEl.className = className || '';
    }

    // Auto-add Player 2 after a delay
    setTimeout(function () {
        if (gameInitialized && iframe.contentWindow.haxRoom) {
            console.log('Note: Open game in iframe to add Player 2, or enable AI bot');
        }
    }, 5000);

})();

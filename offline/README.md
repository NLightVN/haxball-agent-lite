# Haxball Agent Lite - Offline Version

⚽ **Fully offline 2-player Haxball game with Agent API for AI bot development**

Based on [Wazarr94's Haxball Clone](https://github.com/Wazarr94/Wazarr94.github.io) - **same physics as original Haxball!**

---

## 🚀 Quick Start

1. Open `index.html` in your browser
2. Game loads immediately - no server needed!
3. Open browser console (F12) to use Agent API

---

## 🎮 Controls

**Player 1:**
- Arrow Keys: Move
- X: Kick

Add more players by uncommenting code in `script.js` (line 382-400)

---

## 🤖 Agent API Usage

### Get Game State

```javascript
const state = AgentAPI.getState();
console.log(state);

// Returns:
{
    ball: {
        x, y, xspeed, yspeed, radius
    },
    players: [{
        name, team, avatar, disc: {x, y, xspeed, yspeed}, inputs, bot
    }],
    score: {
        red, blue, time, timeLimit, scoreLimit
    },
    stadium: {name, width, height},
    frame: currentFrame
}
```

### Control a Player

```javascript
AgentAPI.setPlayerInput(playerIndex, {
    up: false,
    down: false,
    left: true,
    right: false,
    kick: true
});
```

### Create a Bot

```javascript
// Use EnhancedBot class
const bot = new EnhancedBot(0); // Control player at index 0

// Run every frame
setInterval(() => bot.update(), 1000/60);
```

---

## 📁 Files

- `index.html` - Main game
- `script.js` - Physics engine (Authentic Haxball physics!)
- `bot.js` - Original bot functions
- `agent-api.js` - ⭐ Agent API wrapper
- `enhanced-bot.js` - ⭐ Example bot

---

## 🧠 Creating Custom Bots

```javascript
class MyBot {
    constructor(playerIndex) {
        this.playerIndex = playerIndex;
    }
    
    update() {
        const state = AgentAPI.getState();
        const me = state.players[this.playerIndex];
        const ball = state.ball;
        
        // Your AI logic here
        const input = {
            left: ball.x < me.disc.x,
            right: ball.x > me.disc.x,
            kick: AgentAPI.distance(me.disc, ball) < 25
        };
        
        AgentAPI.setPlayerInput(this.playerIndex, input);
    }
}

const bot = new MyBot(0);
setInterval(() => bot.update(), 1000/60);
```

---

**Ready to build AI agents!** 🤖⚽

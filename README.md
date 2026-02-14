# Haxball Agent Lite ⚽🤖

**Fully offline 2-player Haxball with Agent API for AI bot development**

Based on [Wazarr94's Haxball Clone](https://github.com/Wazarr94/Wazarr94.github.io) - **authentic Haxball physics!**

---

## 🚀 Quick Start

1. **Open `index.html`** in your browser
2. Game starts immediately with 2 players!
3. **Press F12** → Console to use Agent API

No server, no dependencies, 100% offline!

---

## 🎮 Controls

| Player | Team | Controls |
|--------|------|----------|
| **Player 1** | 🔴 Red | **WASD** + **Space** |
| **Player 2** | 🔵 Blue | **Arrow Keys** + **X** |

---

## 🤖 Agent API

### Get Game State

```javascript
const state = AgentAPI.getState();
// Returns: {ball, players, score, stadium, frame}
```

### Control a Player

```javascript
AgentAPI.setPlayerInput(0, {
    up: false,
    down: false,
    left: true,
    right: false,
    kick: true
});
```

### Create a Bot

```javascript
const bot = new EnhancedBot(1); // Control Player 2
setInterval(() => bot.update(), 1000/60); // Run at 60 FPS
```

---

## 📚 Full Documentation

See [`README_FULL.md`](offline/README.md) for:
- Complete API documentation
- Advanced bot examples
- Physics constants
- Custom stadium creation

---

## 📁 Project Structure

```
haxball-agent-lite/
├── index.html          # Main game (LITE version)
├── script.js           # Physics engine (86KB - authentic!)
├── bot.js              # Original bot functions
├── agent-api.js        # Agent API wrapper
├── enhanced-bot.js     # Example AI bot
├── style.css, audio/   # Assets
│
├── offline/            # Original offline version
├── legacy/             # Old experimental versions
└── README.md           # This file
```

---

## ✨ Features

✅ **100% Offline** - No internet required  
✅ **Authentic Physics** - Same as original Haxball  
✅ **Full Agent API** - Complete game state access  
✅ **Easy Bot Development** - Simple JavaScript interface  
✅ **Classic Map** - Official Haxball stadium  
✅ **Clean UI** - Just score and timer  

---

## 🎯 Next Steps

1. **Play**: Open `index.html` and play with a friend
2. **Bot**: F12 console → `new EnhancedBot(1)` to add AI
3. **Customize**: Edit `script.js` to change physics or add features
4. **Train AI**: Use Agent API to build ML-powered bots

---

## 📖 Examples

### Simple Chase Bot

```javascript
function chaseBot() {
    const state = AgentAPI.getState();
    const me = state.players[1]; // Player 2
    const ball = state.ball;
    
    AgentAPI.setPlayerInput(1, {
        left: ball.x < me.disc.x,
        right: ball.x > me.disc.x,
        up: ball.y < me.disc.y,
        down: ball.y > me.disc.y,
        kick: AgentAPI.distance(me.disc, ball) < 30
    });
}

setInterval(chaseBot, 1000/60);
```

---

**Ready to build AI agents!** 🤖⚽

Made with ❤️ for offline Haxball AI development

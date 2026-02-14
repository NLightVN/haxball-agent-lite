# Haxball Agent Lite

Offline 2-player Haxball game using the official Haxball Headless API, designed for AI agent development and testing.

## Features

- ✅ **Authentic Haxball physics** - Uses official Haxball Headless API
- ✅ **2-player local gameplay** - Play on the same computer
- ✅ **Classic maps** - Includes official Haxball stadiums
- ✅ **Customizable controls** - Configure keyboard bindings
- ✅ **Agent-ready** - Built-in API for AI bot development

## Quick Start

### Option 1: Open in Browser (Recommended for testing)

1. Open `index.html` in your browser
2. The game will load automatically
3. Player 1 uses **W, A, S, D, Space**
4. Player 2 uses **Arrow keys, Enter**

### Option 2: Run with Local Server

```bash
npm install
npm start
```

Then open http://localhost:8080

## Controls

### Player 1 (Red Team)
- **W** - Move up
- **A** - Move left
- **S** - Move down
- **D** - Move right
- **Space** - Kick

### Player 2 (Blue Team)
- **↑** - Move up
- **←** - Move left
- **↓** - Move down
- **→** - Move right
- **Enter** - Kick

## AI Agent Development

The game exposes a simple API for creating AI agents:

```javascript
// Get current game state
const state = GameAPI.getState();
// Returns: { ball: {x, y, xspeed, yspeed}, players: [...], score: {...} }

// Control agent (Player 2)
GameAPI.setAgentInput({
    up: true,
    down: false,
    left: false,
    right: true,
    kick: false
});
```

See `src/agent/example-bot.js` for a simple AI implementation.

## Project Structure

```
haxball-agent-lite/
├── index.html              # Main HTML wrapper
├── src/
│   ├── game.js            # Game initialization & controls
│   ├── maps/              # Stadium definitions
│   │   └── classic.js     # Classic Haxball map
│   └── agent/             # AI agent interface
│       ├── agent-api.js   # Agent API
│       └── example-bot.js # Example AI bot
└── README.md
```

## How It Works

This project uses the [Haxball Headless Host API](https://github.com/haxball/haxball-issues/wiki/Headless-Host) to create a local game room that runs entirely in your browser. The physics engine, rendering, and game logic are all provided by the official Haxball implementation.

## License

MIT

// Classic Haxball Stadium Map
// Exported from official Haxball stadium format

const classicStadium = {
    "name": "Classic",
    "width": 420,
    "height": 200,
    "spawnDistance": 180,
    "bg": {
        "type": "grass",
        "width": 370,
        "height": 170,
        "kickOffRadius": 75,
        "cornerRadius": 0
    },
    "vertexes": [
        { "x": -370, "y": 170, "trait": "ballArea", "cMask": ["ball"] },
        { "x": -370, "y": 64, "trait": "ballArea", "cMask": ["ball"] },
        { "x": -370, "y": -64, "trait": "ballArea", "cMask": ["ball"] },
        { "x": -370, "y": -170, "trait": "ballArea", "cMask": ["ball"] },
        { "x": 370, "y": 170, "trait": "ballArea", "cMask": ["ball"] },
        { "x": 370, "y": 64, "trait": "ballArea", "cMask": ["ball"] },
        { "x": 370, "y": -64, "trait": "ballArea", "cMask": ["ball"] },
        { "x": 370, "y": -170, "trait": "ballArea", "cMask": ["ball"] },
        { "x": 0, "y": 200, "trait": "kickOffBarrier" },
        { "x": 0, "y": 75, "trait": "kickOffBarrier" },
        { "x": 0, "y": -75, "trait": "kickOffBarrier" },
        { "x": 0, "y": -200, "trait": "kickOffBarrier" },
        { "x": -380, "y": -64, "trait": "goalNet", "cMask": ["ball"] },
        { "x": -400, "y": -44, "trait": "goalNet", "cMask": ["ball"] },
        { "x": -400, "y": 44, "trait": "goalNet", "cMask": ["ball"] },
        { "x": -380, "y": 64, "trait": "goalNet", "cMask": ["ball"] },
        { "x": 380, "y": -64, "trait": "goalNet", "cMask": ["ball"] },
        { "x": 400, "y": -44, "trait": "goalNet", "cMask": ["ball"] },
        { "x": 400, "y": 44, "trait": "goalNet", "cMask": ["ball"] },
        { "x": 380, "y": 64, "trait": "goalNet", "cMask": ["ball"] }
    ],
    "segments": [
        { "v0": 0, "v1": 1, "trait": "ballArea" },
        { "v0": 2, "v1": 3, "trait": "ballArea" },
        { "v0": 4, "v1": 5, "trait": "ballArea" },
        { "v0": 6, "v1": 7, "trait": "ballArea" },
        { "v0": 12, "v1": 13, "trait": "goalNet", "curve": -90 },
        { "v0": 13, "v1": 14, "trait": "goalNet" },
        { "v0": 14, "v1": 15, "trait": "goalNet", "curve": -90 },
        { "v0": 16, "v1": 17, "trait": "goalNet", "curve": 90 },
        { "v0": 17, "v1": 18, "trait": "goalNet" },
        { "v0": 18, "v1": 19, "trait": "goalNet", "curve": 90 },
        { "v0": 8, "v1": 9, "trait": "kickOffBarrier", "vis": false, "cMask": ["red", "blue"] },
        { "v0": 10, "v1": 11, "trait": "kickOffBarrier", "vis": false, "cMask": ["red", "blue"] }
    ],
    "goals": [
        { "p0": [-370, 64], "p1": [-370, -64], "team": "red" },
        { "p0": [370, 64], "p1": [370, -64], "team": "blue" }
    ],
    "discs": [
        { "pos": [-370, 64], "trait": "goalPost", "color": "FFCCCC" },
        { "pos": [-370, -64], "trait": "goalPost", "color": "FFCCCC" },
        { "pos": [370, 64], "trait": "goalPost", "color": "CCCCFF" },
        { "pos": [370, -64], "trait": "goalPost", "color": "CCCCFF" }
    ],
    "planes": [
        { "normal": [0, 1], "dist": -170, "trait": "ballArea" },
        { "normal": [0, -1], "dist": -170, "trait": "ballArea" },
        { "normal": [0, 1], "dist": -200, "cMask": ["ball"] },
        { "normal": [0, -1], "dist": -200, "cMask": ["ball"] },
        { "normal": [1, 0], "dist": -420, "cMask": ["ball"] },
        { "normal": [-1, 0], "dist": -420, "cMask": ["ball"] }
    ],
    "traits": {
        "ballArea": { "vis": true, "bCoef": 1, "cMask": ["ball"] },
        "goalNet": { "vis": true, "bCoef": 0.1, "cMask": ["ball"] },
        "goalPost": { "radius": 8, "invMass": 0, "bCoef": 0.5 },
        "kickOffBarrier": { "vis": false, "bCoef": 0.1, "cGroup": ["ball"], "cMask": ["red", "blue"] }
    },
    "playerPhysics": {
        "bCoef": 0.5,
        "invMass": 0.5,
        "damping": 0.96,
        "acceleration": 0.1,
        "kickingAcceleration": 0.07,
        "kickingDamping": 0.96,
        "kickStrength": 5
    },
    "ballPhysics": {
        "radius": 10,
        "bCoef": 0.5,
        "invMass": 1,
        "damping": 0.99,
        "color": "FFFFFF",
        "cMask": ["all"],
        "cGroup": ["ball"]
    }
};

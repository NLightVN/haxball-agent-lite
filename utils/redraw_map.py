import json

# Define colors
COLOR_BG = "343a45"
COLOR_LINE = "C7CDD6"
COLOR_GOAL_L = "f08a2b"
COLOR_GOAL_R = "C7CDD6" # Blue side uses white/grey lines for goal
COLOR_KO_BARRIER = "a8b4bd"

# Init empty map structure
data = {
    "name": "Winky's Futsal - Rectangle [ʜᴀxᴍᴏᴅs.ᴄᴏᴍ]",
    "width": 430,
    "height": 200,
    "spawnDistance": 180,
    "bg": { "type": "", "color": COLOR_BG, "width": 0, "height": 0 },
    "playerPhysics": {
        "bCoef": 0.35, "acceleration": 0.11, "kickingAcceleration": 0.083, "kickStrength": 4.65, "radius": 15
    },
    "ballPhysics": {
        "radius": 6.5, "bCoef": 0.465, "invMass": 1.4, "color": "f0bb28"
    },
    "traits": {
        "ballArea": { "vis": True, "bCoef": 1, "cMask": ["ball"], "color": COLOR_LINE },
        "goalNet": { "vis": True, "bCoef": 0.1, "cMask": ["ball"], "color": COLOR_LINE },
        "goalPost": { "radius": 8, "invMass": 0, "bCoef": 0.5 },
        "kickOffBarrier": { "vis": False, "bCoef": 0.1, "cGroup": ["redKO", "blueKO"], "cMask": ["red", "blue"] },
        "none": { "vis": False, "bCoef": 0, "cMask": [] }
    }
}

vertexes = []
segments = []
planes = []
goals = []
discs = []

# --- 1. Field Boundaries & Physical Walls ---
# Dimensions
W, H = 368, 171
GOAL_W, GOAL_H = 400, 65

# Physical Planes (Invisible walls)
planes.append({"normal": [0, 1], "dist": -H, "trait": "ballArea"})
planes.append({"normal": [0, -1], "dist": -H, "trait": "ballArea"})
planes.append({"normal": [1, 0], "dist": -430, "bCoef": 0.1})
planes.append({"normal": [-1, 0], "dist": -430, "bCoef": 0.1})
planes.append({"normal": [0, 1], "dist": -200, "bCoef": 0.1})
planes.append({"normal": [0, -1], "dist": -200, "bCoef": 0.1})
planes.append({"normal": [1, 0], "dist": -GOAL_W, "bCoef": 0.1, "cMask": ["ball"]})
planes.append({"normal": [-1, 0], "dist": -GOAL_W, "bCoef": 0.1, "cMask": ["ball"]})

# --- 2. Visual Lines (Segments) ---

# Helper to add vertex and get index
def add_v(x, y, trait="ballArea", color=None, curve=0):
    v = {"x": x, "y": y, "trait": trait}
    if color: v["color"] = color
    if curve != 0: v["curve"] = curve # Vertex curve param if needed (usually on leg)
    vertexes.append(v)
    return len(vertexes) - 1

def add_s(v0, v1, trait="ballArea", color=None, curve=0):
    s = {"v0": v0, "v1": v1, "trait": trait}
    if color: s["color"] = color
    if curve != 0: s["curve"] = curve
    segments.append(s)

# A. Field Perimeter (Top/Bottom/Side Lines)
# Top Line
v_tl = add_v(-W, -H, "ballArea", COLOR_LINE)
v_tr = add_v(W, -H, "ballArea", COLOR_LINE)
add_s(v_tl, v_tr, "ballArea", COLOR_LINE)

# Bottom Line
v_bl = add_v(-W, H, "ballArea", COLOR_LINE)
v_br = add_v(W, H, "ballArea", COLOR_LINE)
add_s(v_bl, v_br, "ballArea", COLOR_LINE)

# Left Side (Goal Line + Goal Box)
v_gl_t = add_v(-W, -GOAL_H, "ballArea", COLOR_LINE) # Goal line top
v_gl_b = add_v(-W, GOAL_H, "ballArea", COLOR_LINE) # Goal line bottom
add_s(v_tl, v_gl_t, "ballArea", COLOR_LINE) # Line from corner to post
add_s(v_bl, v_gl_b, "ballArea", COLOR_LINE) # Line from corner to post

# Right Side
v_gr_t = add_v(W, -GOAL_H, "ballArea", COLOR_LINE)
v_gr_b = add_v(W, GOAL_H, "ballArea", COLOR_LINE)
add_s(v_tr, v_gr_t, "ballArea", COLOR_LINE)
add_s(v_br, v_gr_b, "ballArea", COLOR_LINE)

# B. Rectangular Goals
# Left Goal (Orange)
v_lg_tl = add_v(-GOAL_W, -GOAL_H, "goalNet", COLOR_GOAL_L) # Back top
v_lg_bl = add_v(-GOAL_W, GOAL_H, "goalNet", COLOR_GOAL_L) # Back bottom
add_s(v_gl_t, v_lg_tl, "goalNet", COLOR_GOAL_L) # Top bar
add_s(v_gl_b, v_lg_bl, "goalNet", COLOR_GOAL_L) # Bottom bar
add_s(v_lg_tl, v_lg_bl, "goalNet", COLOR_GOAL_L) # Back bar

# Right Goal (White/Grey)
v_rg_tr = add_v(GOAL_W, -GOAL_H, "goalNet", COLOR_GOAL_R)
v_rg_br = add_v(GOAL_W, GOAL_H, "goalNet", COLOR_GOAL_R)
add_s(v_gr_t, v_rg_tr, "goalNet", COLOR_GOAL_R)
add_s(v_gr_b, v_rg_br, "goalNet", COLOR_GOAL_R)
add_s(v_rg_tr, v_rg_br, "goalNet", COLOR_GOAL_R)

# C. Half-way Line
v_mid_t = add_v(0, -H, "ballArea", COLOR_LINE)
v_mid_b = add_v(0, H, "ballArea", COLOR_LINE)
add_s(v_mid_t, v_mid_b, "ballArea", COLOR_LINE)

# D. Center Circle (Split Color)
# We use segments with curve to make circle. 2 segments, 180 deg each.
v_cc_t = add_v(0, -65, "none") 
v_cc_b = add_v(0, 65, "none")
add_s(v_cc_t, v_cc_b, "none", COLOR_GOAL_L, 180) # Left half (Orange)
add_s(v_cc_b, v_cc_t, "none", COLOR_LINE, 180)   # Right half (White)

# E. Penalty Arcs
# Left Arc
v_pa_l_t = add_v(-W, -65, "none") # Reuse indices if strict but new is fine
v_pa_l_b = add_v(-W, 65, "none")
add_s(v_pa_l_t, v_pa_l_b, "none", COLOR_LINE, 180) # Curve out

# Right Arc
v_pa_r_t = add_v(W, -65, "none") 
v_pa_r_b = add_v(W, 65, "none")
add_s(v_pa_r_b, v_pa_r_t, "none", COLOR_LINE, 180) # Curve out

# --- 3. Functional Objects ---

# Kickoff Barriers (Invisible)
v_ko_t = add_v(0, 80, "kickOffBarrier")
v_ko_b = add_v(0, -80, "kickOffBarrier")
v_ko_c = add_v(0, 0, "kickOffBarrier") # Center is just approximate, line segments matter
add_s(v_cc_t, v_ko_b, "kickOffBarrier") # From 0,-65 to 0,-80
add_s(v_cc_b, v_ko_t, "kickOffBarrier") # From 0,65 to 0,80
# Center circle barrier
add_s(v_cc_t, v_cc_b, "kickOffBarrier", None, 180)
add_s(v_cc_b, v_cc_t, "kickOffBarrier", None, 180)

# Goals (Scoring Areas)
goals.append({"p0": [W, 65], "p1": [W, -65], "team": "blue", "color": "blue"})
goals.append({"p0": [-W, -65], "p1": [-W, 65], "team": "red", "color": "red"})

# Goal Posts (Discs)
discs.append({"pos": [-W, 65], "trait": "goalPost", "color": COLOR_GOAL_L})
discs.append({"pos": [-W, -65], "trait": "goalPost", "color": COLOR_GOAL_L})
discs.append({"pos": [W, 65], "trait": "goalPost", "color": COLOR_GOAL_R})
discs.append({"pos": [W, -65], "trait": "goalPost", "color": COLOR_GOAL_R})

# Winky's Logo (Middle W) - Simplified geometry
# W shape lines
v_w1 = add_v(-12, -12, "none", COLOR_GOAL_L); v_w2 = add_v(-4, 12, "none", COLOR_GOAL_L); add_s(v_w1, v_w2, "none", COLOR_GOAL_L)
v_w3 = add_v(-4, 12, "none", COLOR_LINE); v_w4 = add_v(4, -12, "none", COLOR_LINE); add_s(v_w3, v_w4, "none", COLOR_LINE)
v_w5 = add_v(4, -12, "none", COLOR_GOAL_L); v_w6 = add_v(12, 12, "none", COLOR_GOAL_L); add_s(v_w5, v_w6, "none", COLOR_GOAL_L)
v_w7 = add_v(12, 12, "none", COLOR_LINE); v_w8 = add_v(20, -12, "none", COLOR_LINE); add_s(v_w7, v_w8, "none", COLOR_LINE)
# Center dot
discs.append({"pos": [0, 0], "radius": 2, "color": "f0bb28", "invMass": 0})

# Assemble
data["vertexes"] = vertexes
data["segments"] = segments
data["planes"] = planes
data["goals"] = goals
data["discs"] = discs

# Write
with open("maps/futsal-fixed.hbs", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Map redrawn successfully")

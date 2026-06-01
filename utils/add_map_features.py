import json
import math

# Load current map
try:
    with open('maps/futsal-fixed.hbs', 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading map: {e}")
    exit(1)

COLOR_LINE = "C7CDD6"
COLOR_GOAL_L = "f08a2b"
COLOR_GOAL_R = "C7CDD6"
W, H = 368, 171
GOAL_W, GOAL_H = 400, 65
ARC_RADIUS = 100 # Adjusted for smaller penalty box (approx 6m)

# Helper to find vertex index by coordinates
def find_v(x, y):
    for i, v in enumerate(data['vertexes']):
        if v.get('x') == x and v.get('y') == y:
            return i
    return -1

def add_v(x, y, trait="ballArea", color=None):
    v = {"x": x, "y": y, "trait": trait}
    if color: v["color"] = color
    data['vertexes'].append(v)
    return len(data['vertexes']) - 1

def add_s(v0, v1, trait="ballArea", color=None, curve=0, vis=None):
    s = {"v0": v0, "v1": v1, "trait": trait}
    if color: s["color"] = color
    if curve != 0: s["curve"] = curve
    if vis is not None: s["vis"] = vis
    data['segments'].append(s)

# 1. REMOVE OLD PENALTY ARCS
# Identify them by their curve property and connection to goal areas
# Or simplier: Remove ANY segment that is an arc (curve != 0) AND is white AND is near goals
# Be careful not to remove center circle or corner arcs if any
# Let's filter segments: Keep if NOT (curve != 0 and near goals)
new_segments = []
for s in data['segments']:
    # Check if it's the old penalty arc we just added
    # They were added at the end of the list usually, but let's be safe
    # We identify them by logic: 180 curve near +/- W
    is_old_arc = False
    if s.get('curve') == 180 or s.get('curve') == -180:
        v0 = data['vertexes'][s['v0']]
        v1 = data['vertexes'][s['v1']]
        # Check if near goal lines (x approx +/- 368)
        if abs(v0['x']) >= 360 and abs(v1['x']) >= 360:
             is_old_arc = True
    
    if not is_old_arc:
        new_segments.append(s)

data['segments'] = new_segments

# 2. ADD NEW PENALTY ARCS (Smaller)
# Left Arc
v_pal_t = add_v(-W, -ARC_RADIUS, "none")
v_pal_b = add_v(-W, ARC_RADIUS, "none")
add_s(v_pal_t, v_pal_b, "none", COLOR_LINE, 180, vis=True)

# Right Arc
v_par_t = add_v(W, -ARC_RADIUS, "none")
v_par_b = add_v(W, ARC_RADIUS, "none")
add_s(v_par_b, v_par_t, "none", COLOR_LINE, 180, vis=True) # Curve reversed for right side


# 3. ADD CENTER VERTICAL LINES connecting Circle to Border
# Top Line: From Center Top (-65) to Border Top (-171)
v_cc_t = find_v(0, -65)
if v_cc_t == -1: v_cc_t = add_v(0, -65, "none") # Should exist from center circle

v_border_t = add_v(0, -H, "none") # Top Border center
add_s(v_cc_t, v_border_t, "none", COLOR_LINE, vis=True)

# Bottom Line: From Center Bottom (65) to Border Bottom (171)
v_cc_b = find_v(0, 65)
if v_cc_b == -1: v_cc_b = add_v(0, 65, "none")

v_border_b = add_v(0, H, "none") # Bottom Border center
add_s(v_cc_b, v_border_b, "none", COLOR_LINE, vis=True)


# Save
with open('maps/futsal-fixed.hbs', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Refined map markings successfully")

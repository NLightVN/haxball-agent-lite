import json

# Load current map
try:
    with open('maps/futsal-fixed.hbs', 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading map: {e}")
    exit(1)

COLOR_LINE = "C7CDD6"
W, H = 368, 171
# User requested 76.
LINE_START_Y = 76

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

# 1. CLEANUP ALL CENTER VERTICAL SEGMENTS (Visual + Invisible Barriers)
# Strategy: 
# - Identify segments where both v0 and v1 have x approx 0.
# - Identify segments that use 'kickOffBarrier' trait.
# - Identify segments that connect to y approx +/- 171 (border) from y approx +/- 60-100 (center).

new_segments = []
for s in data['segments']:
    try:
        v0 = data['vertexes'][s['v0']]
        v1 = data['vertexes'][s['v1']]
        
        # Condition 1: Vertical line at x=0
        is_center_vertical = (abs(v0['x']) < 1 and abs(v1['x']) < 1)
        
        # Condition 2: Uses kickOffBarrier trait (or connected to old kickoff usage)
        is_barrier = (s.get('trait') == 'kickOffBarrier')

        # Logic to remove:
        # If it is a center vertical line (visual or barrier) connecting near center to border, remove it.
        if is_center_vertical:
            y0, y1 = sorted([abs(v0['y']), abs(v1['y'])])
            # Check if it spans from center area (>50) to border area (>150)
            if y0 > 50 and y1 > 150:
                print(f"Removing segment: {s} coords: {v0['y']} to {v1['y']}")
                continue # SKIP adding this segment to new_segments
        
        if is_barrier:
             # Also remove any rogue kickOffBarrier segments even if coords are slightly off
             # But be careful not to remove OTHER barriers if they exist (unlikely in this map)
             print(f"Removing barrier segment: {s}")
             continue
             
        new_segments.append(s)
    except IndexError:
        pass # Handle potential bad indices if any

data['segments'] = new_segments

# 2. ADD SINGLE UNIFIED CENTER LINE (Visual + Barrier) at 76
# Top Line
v_line_t_start = add_v(0, -LINE_START_Y, "kickOffBarrier") 
v_border_t = add_v(0, -H, "kickOffBarrier")
add_s(v_line_t_start, v_border_t, "kickOffBarrier", COLOR_LINE, vis=True)

# Bottom Line
v_line_b_start = add_v(0, LINE_START_Y, "kickOffBarrier")
v_border_b = add_v(0, H, "kickOffBarrier")
add_s(v_line_b_start, v_border_b, "kickOffBarrier", COLOR_LINE, vis=True)

# 3. VERIFY KICKOFF BARRIER TRAIT
# Ensure it blocks players (cGroup/cMask)
if 'kickOffBarrier' not in data['traits']:
    data['traits']['kickOffBarrier'] = {}

# Force correct physics for barrier
data['traits']['kickOffBarrier']['vis'] = True # We handle visibility in segment, but trait default can be true too
data['traits']['kickOffBarrier']['cGroup'] = ["redKO", "blueKO"]
data['traits']['kickOffBarrier']['cMask'] = ["red", "blue"]
data['traits']['kickOffBarrier']['bCoef'] = 0.1 # Bounce coef

# Save
with open('maps/futsal-fixed.hbs', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Cleaned and reset center lines to 76 (barrier+visual)")

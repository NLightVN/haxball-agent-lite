import json
import math

def process_map():
    with open("input_map.hbs", "r") as f:
        data = json.load(f)

    data["name"] = data["name"] + " (A1 Size)"
    
    # Update physics to match A1 precisely
    data["playerPhysics"]["bCoef"] = 0.5
    data["playerPhysics"]["kickStrength"] = 4.545
    data["playerPhysics"]["acceleration"] = 0.11
    data["playerPhysics"]["kickingAcceleration"] = 0.083
    data["playerPhysics"]["damping"] = 0.96
    data["playerPhysics"]["invMass"] = 0.5
    data["playerPhysics"]["radius"] = 15.0

    data["ballPhysics"]["bCoef"] = 0.412
    data["ballPhysics"]["radius"] = 5.8
    data["ballPhysics"]["invMass"] = 1.5
    data["ballPhysics"]["damping"] = 0.99
    
    if "goalPost" in data["traits"]:
        data["traits"]["goalPost"]["radius"] = 4.0
        data["traits"]["goalPost"]["bCoef"] = 0.1

    # Map target size
    target_hw = 368.0
    target_hh = 171.0
    target_gy = 64.0
    
    orig_hw = 400.0
    orig_hh = 200.0
    orig_gy = 70.0

    # We will scale all X coordinates by (368/400)
    # We will scale all Y coordinates by (171/200)
    scale_x = target_hw / orig_hw
    scale_y = target_hh / orig_hh

    # Scale vertexes
    if "vertexes" in data:
        for v in data["vertexes"]:
            if "x" in v: v["x"] = round(v["x"] * scale_x, 2)
            if "y" in v: v["y"] = round(v["y"] * scale_y, 2)

    # Scale top-level and bg properties
    if "width" in data: data["width"] = round(data["width"] * scale_x, 2)
    if "height" in data: data["height"] = round(data["height"] * scale_y, 2)
    if "spawnDistance" in data: data["spawnDistance"] = round(data["spawnDistance"] * scale_x, 2)
    
    if "bg" in data:
        if "width" in data["bg"]: data["bg"]["width"] = round(data["bg"]["width"] * scale_x, 2)
        if "height" in data["bg"]: data["bg"]["height"] = round(data["bg"]["height"] * scale_y, 2)

    # Scale goals
    if "goals" in data:
        for g in data["goals"]:
            g["p0"][0] = round(g["p0"][0] * scale_x, 2)
            g["p0"][1] = round(g["p0"][1] * scale_y, 2)
            g["p1"][0] = round(g["p1"][0] * scale_x, 2)
            g["p1"][1] = round(g["p1"][1] * scale_y, 2)

    # Scale discs (poles)
    if "discs" in data:
        for d in data["discs"]:
            if "pos" in d:
                d["pos"][0] = round(d["pos"][0] * scale_x, 2)
                d["pos"][1] = round(d["pos"][1] * scale_y, 2)
                
            # If the disc is a goal post, set its radius to 4
            if d.get("trait") == "goalPost" or d.get("radius") == 5:
                d["radius"] = 4.0
                d["bCoef"] = 0.1

    # Scale planes
    if "planes" in data:
        for p in data["planes"]:
            if "dist" in p:
                if abs(p["normal"][0]) > 0: # vertical plane
                    p["dist"] = round(p["dist"] * scale_x, 2)
                elif abs(p["normal"][1]) > 0: # horizontal plane
                    p["dist"] = round(p["dist"] * scale_y, 2)

    with open("a1_futsal.hbs", "w") as f:
        json.dump(data, f, separators=(',', ':'))
        
    print("Map generated: a1_futsal.hbs")

if __name__ == "__main__":
    process_map()

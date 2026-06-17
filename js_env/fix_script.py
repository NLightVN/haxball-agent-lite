import json
import os

os.chdir(r"e:\haxball-agent-lite\js_env")

with open('a1_futsal.hbs', 'r', encoding='utf-8') as f:
    map_str = f.read()

with open('script.js', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'stadium = JSON.parse(' in line and 'Load default Classic stadium' in lines[i-1]:
        # Wrap it in backticks correctly
        lines[i+1] = f"        `{map_str}`\n"
        break

with open('script.js', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed script.js")

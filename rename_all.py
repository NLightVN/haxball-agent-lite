import os
import glob
import re

# 1. Directories renaming mapping
dirs_to_rename = {
    r"training\2v2 oriented": r"training\2v2 oriented",
    r"models\2v2 oriented": r"models\2v2 oriented",
}

for old, new in dirs_to_rename.items():
    if os.path.exists(old) and not os.path.exists(new):
        print(f"Renaming {old} -> {new}")
        os.rename(old, new)

# We must update references inside the models directory
model_subdirs = {
    r"models\1v1 oriented\a1.0_checkpoints": r"models\1v1 oriented\a1.1_0_checkpoints",
    r"models\2v2 oriented\a3_1_checkpoints": r"models\2v2 oriented\a2_1_1_checkpoints",
    r"models\2v2 oriented\a3_2_checkpoints": r"models\2v2 oriented\a2_2_2_checkpoints",
    r"models\2v2 oriented\a3_3_checkpoints": r"models\2v2 oriented\a2_3_3_checkpoints",
}
for old, new in model_subdirs.items():
    if os.path.exists(old) and not os.path.exists(new):
        print(f"Renaming {old} -> {new}")
        os.rename(old, new)

# 2. Files renaming mapping
files_to_rename = {
    r"training\1v1 oriented\train_a1.0.py": r"training\1v1 oriented\train_a1.1_0.py",
    r"training\1v1 oriented\train_a1.1.py": r"training\1v1 oriented\train_a1.1_1.py",
    r"training\2v2 oriented\train_a3.1.py": r"training\2v2 oriented\train_a2.1_1.py",
    r"training\2v2 oriented\train_a3.2.py": r"training\2v2 oriented\train_a2.2_2.py",
    r"training\2v2 oriented\train_a3.3.py": r"training\2v2 oriented\train_a2.3_3.py",
    r"play_a3.1.py": r"play_a2.1_1.py",
    r"play_a3.2.py": r"play_a2.2_2.py",
    r"play_a3.3.py": r"play_a2.3_3.py",
    r"models\1v1 oriented\a1.0_final": r"models\1v1 oriented\a1.1_0_final",
    r"models\1v1 oriented\a1.1_final.zip": r"models\1v1 oriented\a1.1_1_final.zip",
    r"models\2v2 oriented\a3.1_final.zip": r"models\2v2 oriented\a2.1_1_final.zip",
    r"models\2v2 oriented\a3.2_final.zip": r"models\2v2 oriented\a2.2_2_final.zip",
    r"models\2v2 oriented\a3.3_final.zip": r"models\2v2 oriented\a2.3_3_final.zip",
}

for old, new in files_to_rename.items():
    if os.path.exists(old) and not os.path.exists(new):
        print(f"Renaming {old} -> {new}")
        os.rename(old, new)

# 3. String Replacement across all .py and .json files
replacements = [
    (r"\bA1\.0\b", "A1.1_0"),
    (r"\bA1\.1\b", "A1.1_1"),
    (r"\bA3\.0\b", "A2.0_0"),
    (r"\bA3\.1\b", "A2.1_1"),
    (r"\bA3\.2\b", "A2.2_2"),
    (r"\bA3\.3\b", "A2.3_3"),
    (r"\ba1\.0\b", "a1.1_0"),
    (r"\ba1\.1\b", "a1.1_1"),
    (r"\ba3\.0\b", "a2.0_0"),
    (r"\ba3\.1\b", "a2.1_1"),
    (r"\ba3\.2\b", "a2.2_2"),
    (r"\ba3\.3\b", "a2.3_3"),
    (r"\ba1_0\b", "a1_1_0"),
    (r"\ba1_1\b", "a1_1_1"),
    (r"\ba3_0\b", "a2_0_0"),
    (r"\ba3_1\b", "a2_1_1"),
    (r"\ba3_2\b", "a2_2_2"),
    (r"\ba3_3\b", "a2_3_3"),
    (r"2v2 oriented", "2v2 oriented"),
]

def replace_in_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    new_content = content
    for pattern, repl in replacements:
        new_content = re.sub(pattern, repl, new_content)
    
    if new_content != content:
        print(f"Updated content of {path}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

for root, _, files in os.walk("."):
    # skip venv, .git, .gemini, __pycache__, and models to avoid corrupting zip files
    if any(ignore in root for ignore in [".git", "venv", ".gemini", "__pycache__", "models"]):
        continue
    for file in files:
        if file.endswith(".py") or file.endswith(".json"):
            path = os.path.join(root, file)
            replace_in_file(path)

print("Done.")

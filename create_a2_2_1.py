import os

source_file = r"training\2v2 oriented\train_a2.2_2.py"
target_file = r"training\2v2 oriented\train_a2.2_1.py"

with open(source_file, "r", encoding="utf-8") as f:
    content = f.read()

# Replace references of A2.2_2 / a2_2_2 / A3_2 / a3_2 to A2.2_1 equivalents
content = content.replace("train_a3.2.py", "train_a2.2_1.py")
content = content.replace("a3.2", "a2.2_1")
content = content.replace("A2.2_2", "A2.2_1")
content = content.replace("A3_2", "A2_2_1")
content = content.replace("a3_2", "a2_2_1")
content = content.replace("OpponentPoolA2_2_1", "PoolManagerA2_2_1") # We'll refactor the class manually

# We will just write the basic replaced content, then apply multi-replace for logic
with open(target_file, "w", encoding="utf-8") as f:
    f.write(content)

print("Created train_a2.2_1.py")

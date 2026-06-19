import re

with open("training/env.py", "r", encoding="utf-8") as f:
    content = f.read()

old_logic = r"""                self\.real_pass_active = False # Hit a teammate
                if pid == 0:
                    # Agent \(Tôi\) touched -> reset sequence
                    self\.investment_sequence = \[pid\]
                else:
                    # Teammate touched
                    if pid in self\.investment_sequence:
                        idx = self\.investment_sequence\.index\(pid\)
                        self\.investment_sequence = self\.investment_sequence\[:idx\+1\]
                    else:
                        self\.investment_sequence\.append\(pid\)"""

new_logic = """                self.real_pass_active = False # Hit a teammate
                if pid in self.investment_sequence:
                    # If player was already in sequence, erase everyone after them
                    idx = self.investment_sequence.index(pid)
                    self.investment_sequence = self.investment_sequence[:idx+1]
                else:
                    # If new to sequence, append
                    self.investment_sequence.append(pid)"""

content = re.sub(old_logic, new_logic, content)

with open("training/env.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched investment sequence logic!")

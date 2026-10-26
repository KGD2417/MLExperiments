# =====================================================================
# Experiment 4: To study and implement Decision Tree (ID3 algorithm)
# =====================================================================
# ID3 picks the attribute with the HIGHEST INFORMATION GAIN at every node.
#   Entropy(S)  = -sum(p_i * log2(p_i))       (0 = pure, 1 = 50/50 mixed)
#   Gain(S, A)  = Entropy(S) - sum(|Sv|/|S| * Entropy(Sv))
# Steps:
#   1. If all rows have the same label -> leaf node
#   2. Else find the attribute with max gain -> make it the node
#   3. Split rows by each value of that attribute, repeat on each part
# Pure Python - no external libraries.

import math     # gives us log2()


# ---------------- Graph helper (drawn with characters, no library needed) ----------------
def bar_chart(names, values, width=40, title=""):
    """Horizontal bar chart: bar length is proportional to the value."""
    print("\n" + title)
    biggest = max(abs(v) for v in values) or 1                 # biggest value = full width bar
    for name, value in zip(names, values):
        bar = "#" * round(abs(value) / biggest * width)        # number of # is proportional to value
        print(f"{name:>14} | {bar} {value:.3f}")


# ---------------- Dataset: classic "Play Tennis" ----------------
features = ["Outlook", "Temp", "Humidity", "Wind"]             # attribute names (last column = label)
data = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"],
]


# ---------------- ID3 functions ----------------
def entropy(rows):
    labels = [r[-1] for r in rows]                             # last column is the label (Yes/No)
    total = len(labels)
    ent = 0
    for label in set(labels):                                  # for every distinct label
        p = labels.count(label) / total                        # probability of that label
        ent -= p * math.log2(p)                                # entropy formula
    return ent


def info_gain(rows, col):
    total = len(rows)
    weighted = 0                                               # weighted entropy after splitting
    for value in set(r[col] for r in rows):                    # every value of this attribute
        subset = [r for r in rows if r[col] == value]          # rows having that value
        weighted += len(subset) / total * entropy(subset)      # |Sv|/|S| * Entropy(Sv)
    return entropy(rows) - weighted                            # gain = before - after


def build_tree(rows, cols):
    labels = [r[-1] for r in rows]
    if len(set(labels)) == 1:                                  # all labels same -> pure leaf
        return labels[0]
    if not cols:                                               # no attributes left -> majority label
        return max(set(labels), key=labels.count)

    best = max(cols, key=lambda c: info_gain(rows, c))         # attribute with maximum gain
    tree = {features[best]: {}}                                # node = {attribute: {value: subtree}}
    remaining = [c for c in cols if c != best]                 # don't use the same attribute again
    for value in sorted(set(r[best] for r in rows)):           # one branch per value
        subset = [r for r in rows if r[best] == value]         # rows going down this branch
        tree[features[best]][value] = build_tree(subset, remaining)   # recursion
    return tree


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):                             # leaf -> print the answer
        print(indent + "--> " + tree)
        return
    for attr, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}|-- {attr} = {value}")             # branch
            print_tree(subtree, indent + "|     ")             # go one level deeper


def predict(tree, sample):
    while isinstance(tree, dict):                              # keep going down until a leaf
        attr = next(iter(tree))                                # attribute tested at this node
        value = sample[features.index(attr)]                   # sample's value for that attribute
        tree = tree[attr].get(value, "Unknown")                # follow that branch
    return tree


# ======================== MAIN PROGRAM ========================
print("=" * 60)
print(" EXPERIMENT 4 : Decision Tree (ID3)")
print("=" * 60)

print("\nDataset:")
print("  " + "  ".join(f"{h:<9}" for h in features + ["Play"]))
for r in data:
    print("  " + "  ".join(f"{v:<9}" for v in r))

yes = sum(1 for r in data if r[-1] == "Yes")                   # count of Yes
print(f"\nYes = {yes}, No = {len(data) - yes}")
print(f"Entropy of whole dataset = {entropy(data):.3f}")

gains = [info_gain(data, i) for i in range(len(features))]     # gain of every attribute at the root
for f, g in zip(features, gains):
    print(f"Gain({f}) = {g:.3f}")
bar_chart(features, gains, title="Information Gain at root (biggest bar = root node)")

tree = build_tree(data, list(range(len(features))))
print("\nDecision Tree:")
print_tree(tree)

correct = sum(1 for r in data if predict(tree, r) == r[-1])    # check on training data
print(f"\nTraining accuracy = {correct}/{len(data)}")

# ---------- User input: predict a new day ----------
print("\nEnter a new day to predict (press Enter to use the default).")
default_sample = ["Sunny", "Cool", "High", "Strong"]
sample = []
for i, f in enumerate(features):
    options = sorted(set(r[i] for r in data))                  # allowed values for this attribute
    text = input(f"{f} {options} [default = {default_sample[i]}]: ").strip()
    sample.append(text.capitalize() if text else default_sample[i])   # "sunny" -> "Sunny"
print(f"\nPrediction for {sample}:  Play = {predict(tree, sample)}")

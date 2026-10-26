# =====================================================================
# Experiment 5: To study bagging & boosting techniques and implement AdaBoost
# =====================================================================
# ENSEMBLE LEARNING = combine many weak models into one strong model.
#
# BAGGING (Bootstrap AGGregatING), e.g. Random Forest
#   - Make many random samples of the data WITH replacement (bootstrap)
#   - Train one model on each sample, all in PARALLEL (independent)
#   - Final answer = majority vote          -> reduces VARIANCE (overfitting)
#
# BOOSTING, e.g. AdaBoost, Gradient Boosting, XGBoost
#   - Train models one after another in SEQUENCE
#   - Each new model focuses on the samples the previous ones got WRONG
#   - Final answer = weighted vote          -> reduces BIAS (underfitting)
#
# AdaBoost algorithm (labels must be +1 / -1):
#   1. Give every sample the same weight  w = 1/n
#   2. Train a weak learner (decision stump) with the lowest WEIGHTED error
#   3. alpha = 0.5 * ln((1 - error) / error)         (how much say this stump gets)
#   4. w = w * e^(-alpha * y * h(x)), then normalise so weights sum to 1
#      (wrong samples -> weight goes UP, correct samples -> weight goes DOWN)
#   5. Repeat steps 2-4. Final prediction = sign( sum(alpha * h(x)) )
# Pure Python - no external libraries.

import math     # gives us log() and exp()
import random   # gives us randrange() for bootstrap sampling in bagging


# ---------------- Graph helpers (drawn with characters, no library needed) ----------------
def scatter_plot(points, lines=(), width=60, height=20, title=""):
    """points = list of (x, y, symbol). lines = list of (function, symbol), drawn as y = function(x)."""
    xs = [p[0] for p in points]                                # collect all x values
    ys = [p[1] for p in points]                                # collect all y values
    pad_x = (max(xs) - min(xs)) * 0.1 or 1                     # 10% empty space on left/right
    pad_y = (max(ys) - min(ys)) * 0.1 or 1                     # 10% empty space on top/bottom
    x_min, x_max = min(xs) - pad_x, max(xs) + pad_x            # x range of the graph
    y_min, y_max = min(ys) - pad_y, max(ys) + pad_y            # y range of the graph
    grid = [[" "] * width for _ in range(height)]              # blank canvas full of spaces

    def to_col(x):                                             # convert an x value to a column number
        return round((x - x_min) / (x_max - x_min) * (width - 1))

    def to_row(y):                                             # convert a y value to a row number (row 0 = top)
        return round((y_max - y) / (y_max - y_min) * (height - 1))

    for f, symbol in lines:                                    # draw lines first so points stay on top
        for col in range(width):                               # go through every column
            x = x_min + col * (x_max - x_min) / (width - 1)    # x value at this column
            row = to_row(f(x))                                 # row where the line is at this x
            if 0 <= row < height:                              # draw only if it is inside the canvas
                grid[row][col] = symbol
    for x, y, symbol in points:                                # draw every point
        grid[to_row(y)][to_col(x)] = symbol

    print("\n" + title)                                        # graph title
    for r in range(height):                                    # print each row with its y value
        y_value = y_max - r * (y_max - y_min) / (height - 1)
        print(f"{y_value:8.2f} |" + "".join(grid[r]))
    print(" " * 9 + "+" + "-" * width)                         # the x axis
    print(" " * 10 + f"{x_min:<10.2f}" + f"{x_max:>{width - 10}.2f}")   # x range labels


def bar_chart(names, values, width=40, title=""):
    """Horizontal bar chart: bar length is proportional to the value."""
    print("\n" + title)
    biggest = max(abs(v) for v in values) or 1                 # biggest value = full width bar
    for name, value in zip(names, values):
        bar = "#" * round(abs(value) / biggest * width)        # number of # is proportional to value
        print(f"{name:>14} | {bar} {value:.3f}")


# ---------------- Input helpers ----------------
def ask(prompt, default):
    """Ask for one number. Pressing Enter keeps the default value."""
    text = input(f"{prompt} [default = {default}]: ").strip()  # read what the user typed
    try:
        return type(default)(text) if text else default        # convert to same type as default (int/float)
    except ValueError:                                         # user typed something that is not a number
        print("  Invalid input, using default.")
        return default


def read_rows(default):
    """Read rows of numbers (one row per line). Enter on the first line -> default data."""
    print("Press Enter directly to use the default data. Press Enter on an empty line to finish.")
    rows = []                                                  # rows typed by the user
    while True:
        line = input("> ").strip()                             # read one line
        if not line:                                           # empty line -> stop reading
            break
        try:
            rows.append([float(v) for v in line.split()])      # "2 3 -1" -> [2.0, 3.0, -1.0]
        except ValueError:
            print("  Invalid row, skipped.")
    return rows if rows else default                           # nothing typed -> default data


def read_point(prompt, default):
    """Read one point such as '4 4'. Enter -> default."""
    text = input(f"{prompt} [default = {' '.join(str(v) for v in default)}]: ").strip()
    try:
        return [float(v) for v in text.split()] if text else default
    except ValueError:
        print("  Invalid input, using default.")
        return default


# ---------------- Default dataset ----------------
# Each row: x1, x2, label (+1 or -1)
DEFAULT_DATA = [[1, 2, -1], [2, 1, -1], [2, 3, -1], [3, 2, -1], [4, 5, 1], [5, 4, 1],
                [5, 6, 1], [6, 5, 1], [3, 6, 1], [6, 2, -1], [1, 5, -1], [7, 7, 1]]


# ---------------- Weak learner: Decision Stump ----------------
# A stump is a tree with ONE question:  "is feature >= threshold ?"
# polarity decides which side gets +1
def stump_predict(x, feature, threshold, polarity):
    return polarity if x[feature] >= threshold else -polarity


def best_stump(X, y, w):
    best, best_err = None, float("inf")                        # nothing found yet
    for feature in range(len(X[0])):                           # try every feature
        for threshold in sorted(set(x[feature] for x in X)):   # try every value as a threshold
            for polarity in (1, -1):                           # try both directions
                err = sum(w[i] for i in range(len(X))          # weighted error = sum of weights of wrong samples
                          if stump_predict(X[i], feature, threshold, polarity) != y[i])
                if err < best_err:                             # keep the stump with the smallest error
                    best_err, best = err, (feature, threshold, polarity)
    return best, best_err


# ---------------- AdaBoost ----------------
def adaboost(X, y, rounds):
    n = len(X)
    w = [1 / n] * n                                            # step 1: equal weights
    model = []                                                 # list of (alpha, stump)
    for r in range(rounds):
        stump, err = best_stump(X, y, w)                       # step 2: best weak learner
        err = min(max(err, 1e-10), 1 - 1e-10)                  # keep error away from 0 and 1 (avoid log(0))
        alpha = 0.5 * math.log((1 - err) / err)                # step 3: stump's importance
        model.append((alpha, stump))
        for i in range(n):                                     # step 4: update sample weights
            w[i] *= math.exp(-alpha * y[i] * stump_predict(X[i], *stump))
        total = sum(w)
        w = [wi / total for wi in w]                           # normalise -> weights add up to 1
        print(f"Round {r + 1}: if x{stump[0] + 1} >= {stump[1]} -> {stump[2]:+d} "
              f"| error = {err:.3f} | alpha = {alpha:.3f}")
    return model, w


def adaboost_predict(model, x):
    score = sum(alpha * stump_predict(x, *stump) for alpha, stump in model)   # weighted vote
    return 1 if score >= 0 else -1                             # sign of the score


# ---------------- Bagging (for comparison) ----------------
def bagging(X, y, n_models, seed=0):
    random.seed(seed)                                          # same random samples every run
    n = len(X)
    models = []
    for _ in range(n_models):
        idx = [random.randrange(n) for _ in range(n)]          # bootstrap: pick n rows WITH replacement
        Xb = [X[i] for i in idx]
        yb = [y[i] for i in idx]
        stump, _ = best_stump(Xb, yb, [1 / n] * n)             # train one stump on this sample
        models.append(stump)
    return models


def bagging_predict(models, x):
    votes = sum(stump_predict(x, *s) for s in models)          # every stump votes +1 or -1
    return 1 if votes >= 0 else -1                             # majority wins


def accuracy(y_true, y_pred):
    return sum(1 for p, t in zip(y_pred, y_true) if p == t) / len(y_true) * 100


# ======================== MAIN PROGRAM ========================
print("=" * 60)
print(" EXPERIMENT 5 : Bagging, Boosting and AdaBoost")
print("=" * 60)
print("Enter data, one row per line:  x1 x2 label(+1/-1)")
data = read_rows(DEFAULT_DATA)
X = [row[:2] for row in data]                                  # features
y = [1 if row[2] > 0 else -1 for row in data]                  # labels; 0 is treated as -1
rounds = ask("Number of AdaBoost rounds", 5)
n_models = ask("Number of models for Bagging", 5)
new_point = read_point("New point to classify as 'x1 x2'", [4, 4])

print("\n=== AdaBoost training ===")
model, final_weights = adaboost(X, y, rounds)
ada_preds = [adaboost_predict(model, x) for x in X]
print("Actual   :", y)
print("Predicted:", ada_preds)

single, _ = best_stump(X, y, [1 / len(X)] * len(X))            # one stump alone, for comparison
single_preds = [stump_predict(x, *single) for x in X]

bag = bagging(X, y, n_models)
bag_preds = [bagging_predict(bag, x) for x in X]

acc_single = accuracy(y, single_preds)
acc_bag = accuracy(y, bag_preds)
acc_ada = accuracy(y, ada_preds)
print(f"\nSingle stump accuracy = {acc_single:.1f}%")
print(f"Bagging accuracy      = {acc_bag:.1f}%")
print(f"AdaBoost accuracy     = {acc_ada:.1f}%")

print(f"\nNew point {new_point}:  AdaBoost -> {adaboost_predict(model, new_point):+d},"
      f"  Bagging -> {bagging_predict(bag, new_point):+d}")

# ---------- Graphs ----------
points = [(x[0], x[1], "+" if label == 1 else "-") for x, label in zip(X, y)]
points.append((new_point[0], new_point[1], "?"))
scatter_plot(points, title="Data:  + = class +1,  - = class -1,  ? = new point")

bar_chart([f"Round {i + 1}" for i in range(len(model))], [a for a, _ in model],
          title="Alpha (importance) of each stump")
bar_chart([f"{x}" for x in X], final_weights,
          title="Final sample weights (big bar = hard sample AdaBoost focused on)")
bar_chart(["Single stump", "Bagging", "AdaBoost"], [acc_single, acc_bag, acc_ada],
          title="Accuracy comparison (%)")

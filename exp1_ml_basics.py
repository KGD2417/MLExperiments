# =====================================================================
# Experiment 1: To study and implement various aspects of Machine Learning
# =====================================================================
# Aspects covered:
#   1. Dataset            -> features (X) and labels (y)
#   2. Statistics         -> mean, variance, standard deviation
#   3. Train / Test split -> learn on one part, check on unseen part
#   4. Normalization      -> Min-Max scaling to the range 0..1
#   5. Model              -> K-Nearest Neighbours (KNN) classifier
#   6. Evaluation         -> confusion matrix, accuracy, precision, recall, F1
#   7. Visualization      -> scatter plot and bar chart (drawn with text)
# Pure Python - no external libraries.

import math     # gives us sqrt() for distance and standard deviation
import random   # gives us shuffle() to mix the data before splitting


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
            rows.append([float(v) for v in line.split()])      # "2 5 1" -> [2.0, 5.0, 1.0]
        except ValueError:
            print("  Invalid row, skipped.")
    return rows if rows else default                           # nothing typed -> default data


def read_point(prompt, default):
    """Read one point such as '6 6'. Enter -> default."""
    text = input(f"{prompt} [default = {' '.join(str(v) for v in default)}]: ").strip()
    try:
        return [float(v) for v in text.split()] if text else default
    except ValueError:
        print("  Invalid input, using default.")
        return default


# ---------------- Default dataset ----------------
# Each row: hours_studied, hours_slept, label (1 = pass, 0 = fail)
DEFAULT_DATA = [
    [1, 4, 0], [2, 5, 0], [3, 6, 0], [4, 5, 0], [5, 7, 1], [6, 8, 1], [7, 6, 1], [8, 7, 1],
    [2, 3, 0], [3, 4, 0], [6, 7, 1], [7, 8, 1], [1, 6, 0], [8, 5, 1], [5, 5, 1], [4, 7, 0],
]


# ---------------- Aspect 3: Train / Test split ----------------
def train_test_split(X, y, test_ratio, seed=1):
    idx = list(range(len(X)))                                  # indexes 0, 1, 2, ..., n-1
    random.seed(seed)                                          # fixed seed -> same split every run
    random.shuffle(idx)                                        # mix the indexes randomly
    n_test = max(1, int(len(X) * test_ratio))                  # how many samples go to the test set (min 1)
    test_idx, train_idx = idx[:n_test], idx[n_test:]           # first part = test, rest = train
    X_train = [X[i] for i in train_idx]                        # pick training features
    X_test = [X[i] for i in test_idx]                          # pick testing features
    y_train = [y[i] for i in train_idx]                        # pick training labels
    y_test = [y[i] for i in test_idx]                          # pick testing labels
    return X_train, X_test, y_train, y_test


# ---------------- Aspect 4: Normalization (Min-Max scaling) ----------------
# x_scaled = (x - min) / (max - min)   -> every feature ends up between 0 and 1
def min_max_fit(X):
    columns = list(zip(*X))                                    # transpose: rows -> columns
    mins = [min(c) for c in columns]                           # minimum of each feature
    maxs = [max(c) for c in columns]                           # maximum of each feature
    return mins, maxs


def min_max_transform(X, mins, maxs):
    scaled = []
    for row in X:                                              # scale each row
        scaled.append([(row[j] - mins[j]) / ((maxs[j] - mins[j]) or 1)   # "or 1" avoids divide by zero
                       for j in range(len(row))])
    return scaled


# ---------------- Aspect 5: Model - K-Nearest Neighbours ----------------
def distance(a, b):
    # Euclidean distance = sqrt((a1-b1)^2 + (a2-b2)^2 + ...)
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def knn_predict(X_train, y_train, point, k):
    pairs = [(distance(x, point), label) for x, label in zip(X_train, y_train)]   # distance to every train point
    pairs.sort()                                               # nearest first
    top_k = [label for _, label in pairs[:k]]                  # labels of the k nearest points
    return max(set(top_k), key=top_k.count)                    # majority vote


# ---------------- Aspect 6: Evaluation ----------------
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)   # true positive: said pass, was pass
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)   # true negative: said fail, was fail
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)   # false positive: said pass, was fail
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)   # false negative: said fail, was pass
    return tp, tn, fp, fn


def metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)                 # correct / total
    precision = tp / (tp + fp) if tp + fp else 0               # of predicted pass, how many really passed
    recall = tp / (tp + fn) if tp + fn else 0                  # of real pass, how many we found
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0   # balance of P and R
    return accuracy, precision, recall, f1


# ======================== MAIN PROGRAM ========================
print("=" * 60)
print(" EXPERIMENT 1 : Various Aspects of Machine Learning")
print("=" * 60)

# ----- Aspect 1: Dataset (from the user or default) -----
print("Enter training data, one row per line:  hours_studied hours_slept label(0=fail,1=pass)")
data = read_rows(DEFAULT_DATA)                                 # list of [studied, slept, label]
X = [row[:2] for row in data]                                  # features = first two columns
y = [int(row[2]) for row in data]                              # label = third column
test_ratio = ask("Test ratio (fraction of data kept for testing)", 0.25)
k = ask("Value of k for KNN", 3)
new_student = read_point("New student to predict as 'hours_studied hours_slept'", [6, 6])
print(f"\nTotal samples = {len(X)},  features = {len(X[0])}")

# ----- Aspect 2: Descriptive statistics -----
print("\n--- Descriptive statistics ---")
for j, name in enumerate(["hours_studied", "hours_slept"]):    # for each feature column
    col = [row[j] for row in X]                                # all values of this feature
    mean = sum(col) / len(col)                                 # mean = sum / count
    var = sum((v - mean) ** 2 for v in col) / len(col)         # variance = average squared distance from mean
    print(f"{name:>14}: mean={mean:.2f}  variance={var:.2f}  std={math.sqrt(var):.2f}")

# ----- Aspect 3: Split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio)
print(f"\nTrain size = {len(X_train)},  Test size = {len(X_test)}")

# ----- Aspect 4: Normalization (fit on TRAIN only, to avoid data leakage) -----
mins, maxs = min_max_fit(X_train)
X_train_n = min_max_transform(X_train, mins, maxs)
X_test_n = min_max_transform(X_test, mins, maxs)
print("\nFirst 3 training rows before -> after scaling:")
for raw, scaled in zip(X_train[:3], X_train_n[:3]):
    print(f"  {raw} -> {[round(v, 2) for v in scaled]}")

# ----- Aspect 5: Model -----
y_pred = [knn_predict(X_train_n, y_train, p, k) for p in X_test_n]   # predict every test point
print("\nActual   :", y_test)
print("Predicted:", y_pred)

# ----- Aspect 6: Evaluation -----
tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                 Predicted 1   Predicted 0")
print(f"   Actual 1        TP = {tp:<5}     FN = {fn}")
print(f"   Actual 0        FP = {fp:<5}     TN = {tn}")
acc, prec, rec, f1 = metrics(tp, tn, fp, fn)
print(f"Accuracy={acc:.2f}  Precision={prec:.2f}  Recall={rec:.2f}  F1={f1:.2f}")

# ----- Predict the user's new student -----
new_scaled = min_max_transform([new_student], mins, maxs)[0]   # scale with the SAME min/max as training
result = knn_predict(X_train_n, y_train, new_scaled, k)
print(f"\nNew student {new_student} -> {'PASS' if result == 1 else 'FAIL'}")

# ----- Aspect 7: Visualization -----
points = [(x[0], x[1], "X" if label == 1 else "o") for x, label in zip(X, y)]   # X = pass, o = fail
points.append((new_student[0], new_student[1], "?"))           # ? = the new student
scatter_plot(points, title="Data  (x-axis = hours studied, y-axis = hours slept)   o = fail, X = pass, ? = new")
bar_chart(["Accuracy", "Precision", "Recall", "F1 score"], [acc, prec, rec, f1], title="Evaluation metrics")

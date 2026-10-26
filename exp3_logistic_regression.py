# =====================================================================
# Experiment 3: To study and implement Logistic Regression
# =====================================================================
# Used for BINARY CLASSIFICATION (answer is 0 or 1).
#   z = w1*x1 + w2*x2 + b                 (linear part)
#   p = sigmoid(z) = 1 / (1 + e^-z)       (squash z into a probability 0..1)
#   predict 1 if p >= 0.5 else 0
# Loss (Log-loss / Binary Cross Entropy):
#   L = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )
# Gradient descent:
#   dw_j = (1/n) * sum((p - y) * x_j),   db = (1/n) * sum(p - y)
# Pure Python - no external libraries.

import math     # gives us exp() and log()


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
            rows.append([float(v) for v in line.split()])      # "2 5 0" -> [2.0, 5.0, 0.0]
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


# ---------------- Model functions ----------------
def sigmoid(z):
    # Written in two ways so math.exp() never overflows for very large |z|
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def predict_prob(x, w, b):
    z = sum(w[j] * x[j] for j in range(len(x))) + b            # linear combination w.x + b
    return sigmoid(z)                                          # convert to probability


def log_loss(X, y, w, b):
    eps = 1e-9                                                 # tiny number so we never take log(0)
    total = 0
    for xi, yi in zip(X, y):
        p = predict_prob(xi, w, b)                             # predicted probability of class 1
        total += -(yi * math.log(p + eps) + (1 - yi) * math.log(1 - p + eps))
    return total / len(X)                                      # average loss


# ======================== MAIN PROGRAM ========================
print("=" * 60)
print(" EXPERIMENT 3 : Logistic Regression")
print("=" * 60)
print("Enter data, one row per line:  x1 x2 label(0/1)   (e.g. studied slept pass)")
data = read_rows(DEFAULT_DATA)
X = [row[:2] for row in data]                                  # features
y = [int(row[2]) for row in data]                              # labels 0 / 1
lr = ask("Learning rate", 0.1)
epochs = ask("Number of epochs", 5000)
new_point = read_point("New point to classify as 'x1 x2'", [6, 6])

# ---------- Training (Gradient Descent) ----------
n, d = len(X), len(X[0])                                       # n samples, d features
w = [0.0] * d                                                  # start with all weights = 0
b = 0.0                                                        # start with bias = 0
loss_history = []                                              # (epoch, loss) for the graph
print()
for epoch in range(epochs):
    dw = [0.0] * d                                             # gradient for each weight
    db = 0.0                                                   # gradient for the bias
    for xi, yi in zip(X, y):
        error = predict_prob(xi, w, b) - yi                    # (p - y) for this sample
        for j in range(d):
            dw[j] += error * xi[j]                             # add this sample's share of the gradient
        db += error
    for j in range(d):
        w[j] -= lr * dw[j] / n                                 # update each weight (average gradient)
    b -= lr * db / n                                           # update bias
    if epoch % max(1, epochs // 50) == 0:                      # save ~50 points for the loss graph
        loss_history.append((epoch, log_loss(X, y, w, b)))
    if epoch % max(1, epochs // 5) == 0:                       # print 5 progress lines
        print(f"Epoch {epoch:5d}: loss = {log_loss(X, y, w, b):.4f}")

print(f"\nWeights = {[round(v, 3) for v in w]},  bias = {b:.3f}")
print(f"Equation: z = {w[0]:.3f}*x1 + {w[1]:.3f}*x2 + {b:.3f}")

# ---------- Evaluation ----------
y_pred = [1 if predict_prob(xi, w, b) >= 0.5 else 0 for xi in X]   # threshold 0.5
correct = sum(1 for a, p in zip(y, y_pred) if a == p)          # how many are right
print("Actual   :", y)
print("Predicted:", y_pred)
print(f"Accuracy = {correct}/{n} = {correct / n * 100:.1f}%")

# ---------- New prediction ----------
p = predict_prob(new_point, w, b)
print(f"\nPoint {new_point}: P(class 1) = {p:.3f}  ->  class {1 if p >= 0.5 else 0}")

# ---------- Graphs ----------
points = [(xi[0], xi[1], "X" if yi == 1 else "o") for xi, yi in zip(X, y)]
points.append((new_point[0], new_point[1], "?"))
lines = []
if w[1] != 0:                                                  # boundary: w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b)/w2
    lines.append((lambda x1: -(w[0] * x1 + b) / w[1], "."))
scatter_plot(points, lines, title="Data: o = class 0, X = class 1, ? = new point, ... = decision boundary")

curve = [(z / 2, sigmoid(z / 2), "*") for z in range(-16, 17)]  # z from -8 to 8
scatter_plot(curve, title="Sigmoid curve: p = 1 / (1 + e^-z)   (x-axis = z, y-axis = p)")

scatter_plot([(e, ls, "*") for e, ls in loss_history], title="Log-loss vs Epoch -> should go DOWN")

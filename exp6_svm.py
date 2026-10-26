# =====================================================================
# Experiment 6: To study and implement Support Vector Machine (Linear SVM)
# =====================================================================
# SVM finds the line (hyperplane)  w.x + b = 0  that separates two classes
# with the MAXIMUM MARGIN. Labels must be +1 / -1.
#   Margin lines :  w.x + b = +1   and   w.x + b = -1
#   Margin width :  2 / ||w||
#   Support vectors = points lying on / closest to the margin lines
#
# Hinge loss: L = lambda*||w||^2 + (1/n) * sum( max(0, 1 - y*(w.x + b)) )
# Gradient descent rule for each sample:
#   if y*(w.x + b) >= 1   (correct and outside margin):  w = w - lr * (2*lambda*w)
#   else                  (inside margin or wrong):      w = w - lr * (2*lambda*w - y*x)
#                                                        b = b + lr * y
# Pure Python - no external libraries.


# ---------------- Graph helper (drawn with characters, no library needed) ----------------
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
# Each row: x1, x2, label (+1 or -1). Two groups that can be separated by a line.
DEFAULT_DATA = [[1, 2, -1], [2, 3, -1], [3, 3, -1], [2, 1, -1], [3, 2, -1],
                [6, 5, 1], [7, 8, 1], [8, 6, 1], [7, 5, 1], [6, 7, 1]]


# ---------------- SVM functions ----------------
def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))             # a1*b1 + a2*b2 + ...


def train_svm(X, y, lr, lam, epochs):
    w = [0.0] * len(X[0])                                      # start with weights = 0
    b = 0.0                                                    # start with bias = 0
    for _ in range(epochs):                                    # repeat over the data many times
        for xi, yi in zip(X, y):                               # look at one sample at a time
            if yi * (dot(w, xi) + b) >= 1:                     # correctly classified, outside margin
                w = [w[j] - lr * (2 * lam * w[j]) for j in range(len(w))]                 # only shrink w (widen margin)
            else:                                              # inside margin or misclassified
                w = [w[j] - lr * (2 * lam * w[j] - yi * xi[j]) for j in range(len(w))]   # push boundary
                b += lr * yi                                   # shift the boundary
    return w, b


def predict(x, w, b):
    return 1 if dot(w, x) + b >= 0 else -1                     # which side of the line is x on?


# ======================== MAIN PROGRAM ========================
print("=" * 60)
print(" EXPERIMENT 6 : Support Vector Machine (Linear)")
print("=" * 60)
print("Enter data, one row per line:  x1 x2 label(+1/-1)")
data = read_rows(DEFAULT_DATA)
X = [row[:2] for row in data]                                  # features
y = [1 if row[2] > 0 else -1 for row in data]                  # labels; 0 is treated as -1
lr = ask("Learning rate", 0.001)
lam = ask("Regularization lambda", 0.01)
epochs = ask("Number of epochs", 2000)
new_point = read_point("New point to classify as 'x1 x2'", [4, 4])

w, b = train_svm(X, y, lr, lam, epochs)
print(f"\nWeights w = [{w[0]:.3f}, {w[1]:.3f}],  bias b = {b:.3f}")
print(f"Hyperplane: {w[0]:.3f}*x1 + {w[1]:.3f}*x2 + ({b:.3f}) = 0")
norm_w = dot(w, w) ** 0.5                                      # ||w|| = length of w
if norm_w:
    print(f"Margin width = 2/||w|| = {2 / norm_w:.3f}")

preds = [predict(x, w, b) for x in X]
print("\nActual   :", y)
print("Predicted:", preds)
acc = sum(1 for p, t in zip(preds, y) if p == t) / len(y) * 100
print(f"Accuracy = {acc:.1f}%")

# Support vectors = points with the smallest y*(w.x+b) (closest to the boundary)
print("\nSupport vectors (closest points to the boundary):")
scores = sorted((yi * (dot(w, xi) + b), xi) for xi, yi in zip(X, y))
for s, xi in scores[:3]:
    print(f"  {xi}   y*(w.x+b) = {s:.3f}")

print(f"\nNew point {new_point} -> class {predict(new_point, w, b):+d}")

# ---------- Graph ----------
points = [(x[0], x[1], "+" if label == 1 else "-") for x, label in zip(X, y)]
points.append((new_point[0], new_point[1], "?"))
lines = []
if w[1] != 0:                                                  # solve w1*x1 + w2*x2 + b = k for x2
    lines.append((lambda x1: (1 - b - w[0] * x1) / w[1], ":"))    # margin line  (+1)
    lines.append((lambda x1: (-1 - b - w[0] * x1) / w[1], ":"))   # margin line  (-1)
    lines.append((lambda x1: (0 - b - w[0] * x1) / w[1], "."))    # hyperplane   (0)
scatter_plot(points, lines,
             title="+ = class +1, - = class -1, ? = new point, ... = hyperplane, ::: = margins")

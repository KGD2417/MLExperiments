# =====================================================================
# Experiment 8: To study and implement PCA (Principal Component Analysis)
# =====================================================================
# PCA = dimensionality reduction. It finds new axes (principal components)
# along which the data varies the MOST, then keeps only the top k of them.
# Steps:
#   1. Centre the data (subtract the mean of each feature)
#   2. Compute the covariance matrix
#   3. Find eigenvalues & eigenvectors of the covariance matrix
#        eigenvector = direction of a principal component
#        eigenvalue  = amount of variance in that direction
#   4. Sort by eigenvalue (largest first)  -> PC1, PC2, ...
#   5. Project the data onto the top k eigenvectors -> reduced data
# Eigenvectors are found with POWER ITERATION:
#   v = M*v, normalise, repeat -> v turns into the biggest eigenvector.
#   Then remove that component (deflation) and repeat for the next one.
# Pure Python - no external libraries.

import math     # gives us sqrt()
import random   # gives us a random starting vector for power iteration


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
            rows.append([float(v) for v in line.split()])      # "2.5 2.4" -> [2.5, 2.4]
        except ValueError:
            print("  Invalid row, skipped.")
    return rows if rows else default                           # nothing typed -> default data


# ---------------- Default dataset ----------------
# Classic PCA example (2 strongly correlated features)
DEFAULT_DATA = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
                [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]]


# ---------------- Matrix helpers ----------------
def mat_vec(M, v):
    # matrix x vector: each output = dot product of a row of M with v
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def normalise(v):
    length = math.sqrt(sum(x * x for x in v))                  # length of the vector
    return [x / length for x in v] if length else v            # make length = 1 (unit vector)


def power_iteration(M, iters=1000):
    random.seed(0)                                             # same start every run
    v = normalise([random.random() + 0.1 for _ in M])          # random start vector
    for _ in range(iters):
        v = normalise(mat_vec(M, v))                           # multiply and normalise again and again
    eigenvalue = sum(a * b for a, b in zip(v, mat_vec(M, v)))  # eigenvalue = v . (M v)
    return eigenvalue, v


def eigen(M):
    M = [row[:] for row in M]                                  # copy so we don't change the original
    values, vectors = [], []
    for _ in range(len(M)):                                    # one eigen pair per dimension
        val, vec = power_iteration(M)
        values.append(val)
        vectors.append(vec)
        # Deflation: subtract  val * v * v^T  so the next run finds the next biggest one
        M = [[M[i][j] - val * vec[i] * vec[j] for j in range(len(M))] for i in range(len(M))]
    return values, vectors                                     # already in order: largest first


# ======================== MAIN PROGRAM ========================
print("=" * 60)
print(" EXPERIMENT 8 : PCA (Principal Component Analysis)")
print("=" * 60)
print("Enter data, one row per line, features separated by spaces (e.g. '2.5 2.4')")
X = read_rows(DEFAULT_DATA)
n, d = len(X), len(X[0])                                       # n samples, d features
k = ask(f"Number of components to keep (1 to {d})", 1)
k = min(max(k, 1), d)                                          # keep k inside the valid range

# ---------- Step 1: Centre the data ----------
means = [sum(row[j] for row in X) / n for j in range(d)]       # mean of each feature
Xc = [[row[j] - means[j] for j in range(d)] for row in X]      # subtract the mean
print("\nMean of each feature:", [round(m, 3) for m in means])

# ---------- Step 2: Covariance matrix ----------
# cov(i, j) = sum((xi - mean_i)(xj - mean_j)) / (n - 1)
cov = [[sum(Xc[r][i] * Xc[r][j] for r in range(n)) / (n - 1) for j in range(d)] for i in range(d)]
print("Covariance matrix:")
for row in cov:
    print("   ", [round(v, 4) for v in row])

# ---------- Step 3 & 4: Eigenvalues and eigenvectors ----------
values, vectors = eigen(cov)
total = sum(values)                                            # total variance in the data
explained = [v / total * 100 for v in values]                  # % variance of each component
print()
for i in range(d):
    print(f"PC{i + 1}: eigenvalue = {values[i]:.4f}, eigenvector = {[round(x, 4) for x in vectors[i]]}, "
          f"variance explained = {explained[i]:.2f}%")
print(f"Keeping {k} component(s) keeps {sum(explained[:k]):.2f}% of the information")

# ---------- Step 5: Project onto the top k components ----------
projected = [[sum(row[j] * vectors[c][j] for j in range(d)) for c in range(k)] for row in Xc]
print(f"\nData reduced from {d}D to {k}D:")
for orig, p in zip(X, projected):
    print(f"  {orig} -> {[round(v, 4) for v in p]}")

# ---------- Graphs ----------
bar_chart([f"PC{i + 1}" for i in range(d)], explained, title="Variance explained by each component (%)")

if d == 2:                                                     # lines can only be drawn for 2D data
    lines = []
    for vec, symbol in zip(vectors, ".:"):                     # PC1 = ...., PC2 = ::::
        if vec[0] != 0:
            slope = vec[1] / vec[0]                            # direction of the component as a slope
            lines.append((lambda x, s=slope: s * x, symbol))
    scatter_plot([(p[0], p[1], "o") for p in Xc], lines,
                 title="Centred data (o) with PC1 (....) and PC2 (::::) directions")

scatter_plot([(p[0], 0, "o") for p in projected], height=5,
             title="Data after reduction: values on the PC1 axis")

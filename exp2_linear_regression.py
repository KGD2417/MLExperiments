# =====================================================================
# Experiment 2: To study and implement Linear Regression
# =====================================================================
# Model: y = m*x + c        (m = slope, c = intercept)
# Method 1: Least Squares formula (direct answer)
#     m = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
#     c = y_mean - m * x_mean
# Method 2: Gradient Descent (learn step by step)
#     Cost (MSE) = (1/n) * sum((y - y_pred)^2)
#     dm = (-2/n) * sum(x * (y - y_pred))
#     dc = (-2/n) * sum(y - y_pred)
#     m = m - lr * dm ,  c = c - lr * dc
# Evaluation: MSE and R^2 score
# Pure Python - no external libraries.

import math     # gives us isnan()/isinf() to detect if gradient descent blows up


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
            rows.append([float(v) for v in line.split()])      # "3 42" -> [3.0, 42.0]
        except ValueError:
            print("  Invalid row, skipped.")
    return rows if rows else default                           # nothing typed -> default data


# ---------------- Default dataset ----------------
# Each row: years_of_experience, salary (in thousands)
DEFAULT_DATA = [[1, 30], [2, 35], [3, 42], [4, 48], [5, 52],
                [6, 60], [7, 65], [8, 70], [9, 78], [10, 83]]


# ======================== MAIN PROGRAM ========================
print("=" * 60)
print(" EXPERIMENT 2 : Linear Regression")
print("=" * 60)
print("Enter data, one row per line:  x y   (e.g. experience salary)")
data = read_rows(DEFAULT_DATA)                                 # list of [x, y]
X = [row[0] for row in data]                                   # input values
Y = [row[1] for row in data]                                   # output (target) values
n = len(X)                                                     # number of samples
lr = ask("Learning rate for gradient descent", 0.01)
epochs = ask("Number of epochs (iterations)", 5000)
new_x = ask("x value to predict", 12.0)

# ---------- Method 1: Least Squares ----------
x_mean = sum(X) / n                                            # average of x
y_mean = sum(Y) / n                                            # average of y
num = sum((X[i] - x_mean) * (Y[i] - y_mean) for i in range(n))  # covariance part (top of formula)
den = sum((X[i] - x_mean) ** 2 for i in range(n))              # variance part (bottom of formula)
m = num / den                                                  # slope
c = y_mean - m * x_mean                                        # intercept
print(f"\nLeast Squares    : y = {m:.3f}x + {c:.3f}")

# ---------- Method 2: Gradient Descent ----------
m_gd, c_gd = 0.0, 0.0                                          # start with slope 0 and intercept 0
cost_history = []                                              # (epoch, cost) pairs for the graph
for epoch in range(epochs):
    y_pred = [m_gd * x + c_gd for x in X]                      # current predictions
    errors = [Y[i] - y_pred[i] for i in range(n)]              # actual - predicted
    cost = sum(e * e for e in errors) / n                      # mean squared error
    if math.isnan(cost) or math.isinf(cost):                   # cost exploded -> learning rate too big
        print("Gradient descent diverged! Try a smaller learning rate (e.g. 0.001).")
        break
    if epoch % max(1, epochs // 50) == 0:                      # save about 50 points for the graph
        cost_history.append((epoch, cost))
    dm = (-2 / n) * sum(X[i] * errors[i] for i in range(n))    # gradient (slope) of cost w.r.t. m
    dc = (-2 / n) * sum(errors)                                # gradient of cost w.r.t. c
    m_gd -= lr * dm                                            # move m opposite to gradient
    c_gd -= lr * dc                                            # move c opposite to gradient
print(f"Gradient Descent : y = {m_gd:.3f}x + {c_gd:.3f}")

# ---------- Evaluation ----------
y_pred = [m * x + c for x in X]                                # predictions using least squares line
ss_res = sum((Y[i] - y_pred[i]) ** 2 for i in range(n))        # error left after fitting the line
ss_tot = sum((Y[i] - y_mean) ** 2 for i in range(n))           # total spread of y around its mean
mse = ss_res / n                                               # mean squared error
r2 = 1 - ss_res / ss_tot if ss_tot else 1                      # R^2: 1.0 = perfect fit
print(f"\nMSE  = {mse:.3f}")
print(f"RMSE = {math.sqrt(mse):.3f}")
print(f"R^2  = {r2:.4f}   (closer to 1 = better fit)")

print("\n    x      actual   predicted")
for i in range(n):
    print(f"{X[i]:6.2f}  {Y[i]:9.2f}  {y_pred[i]:9.2f}")

# ---------- Prediction ----------
print(f"\nPrediction for x = {new_x}  ->  y = {m * new_x + c:.2f}")

# ---------- Graphs ----------
points = [(X[i], Y[i], "o") for i in range(n)]                 # o = actual data
points.append((new_x, m * new_x + c, "?"))                     # ? = new prediction
scatter_plot(points, lines=[(lambda x: m * x + c, ".")],
             title="Data (o) and fitted regression line (.)   ? = prediction")
if cost_history:
    # log10 is used on the y-axis because cost drops very fast at the start (3000 -> 0.5)
    scatter_plot([(e, math.log10(cst + 1e-12), "*") for e, cst in cost_history],
                 title="log10(Cost) vs Epoch  -> should go DOWN as the model learns")

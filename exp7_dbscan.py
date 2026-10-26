# =====================================================================
# Experiment 7: To study and implement DBSCAN
# (Density-Based Spatial Clustering of Applications with Noise)
# =====================================================================
# Unsupervised learning: no labels, the algorithm finds groups by itself.
# Parameters:
#   eps     = radius of the neighbourhood circle around a point
#   min_pts = minimum points (including itself) inside eps to be a CORE point
# Point types:
#   Core   : has >= min_pts points within eps
#   Border : not core, but lies within eps of a core point
#   Noise  : neither core nor border  -> label -1 (outlier)
# Algorithm:
#   For every unvisited point: if it is core, start a new cluster and keep
#   adding all density-reachable points (expand through other core points).
# Pure Python - no external libraries.

import math     # gives us sqrt()


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
            rows.append([float(v) for v in line.split()])      # "1.5 1.2" -> [1.5, 1.2]
        except ValueError:
            print("  Invalid row, skipped.")
    return rows if rows else default                           # nothing typed -> default data


# ---------------- Default dataset ----------------
# Each row: x, y.  Two dense groups + two far away points (noise)
DEFAULT_DATA = [[1, 1], [1.5, 1.2], [1.2, 1.8], [2, 1.5], [1.8, 2],          # group A
                [8, 8], [8.5, 8.2], [8.2, 8.8], [9, 8.5], [8.7, 9],          # group B
                [4.5, 5], [15, 2]]                                           # noise


# ---------------- DBSCAN functions ----------------
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)   # Euclidean distance


def region_query(points, i, eps):
    # indexes of all points within distance eps of point i (includes i itself)
    return [j for j in range(len(points)) if distance(points[i], points[j]) <= eps]


def dbscan(points, eps, min_pts):
    UNVISITED, NOISE = None, -1                                # special label values
    labels = [UNVISITED] * len(points)                         # nobody has a cluster yet
    cluster = 0                                                # cluster counter

    for i in range(len(points)):
        if labels[i] is not UNVISITED:                         # already processed -> skip
            continue
        neighbours = region_query(points, i, eps)              # points near i
        if len(neighbours) < min_pts:                          # not dense enough
            labels[i] = NOISE                                  # mark noise (may become border later)
            continue

        cluster += 1                                           # i is a core point -> new cluster
        labels[i] = cluster
        queue = list(neighbours)                               # points to check for this cluster
        while queue:                                           # expand the cluster
            j = queue.pop(0)                                   # take the next point
            if labels[j] == NOISE:                             # was noise, but is near a core point
                labels[j] = cluster                            # -> becomes a border point
            if labels[j] is not UNVISITED:                     # already has a cluster -> skip
                continue
            labels[j] = cluster                                # add j to the cluster
            j_neighbours = region_query(points, j, eps)
            if len(j_neighbours) >= min_pts:                   # j is also core -> its neighbours join too
                queue.extend(j_neighbours)
    return labels


# ======================== MAIN PROGRAM ========================
print("=" * 60)
print(" EXPERIMENT 7 : DBSCAN Clustering")
print("=" * 60)
print("Enter points, one per line:  x y")
points = read_rows(DEFAULT_DATA)
eps = ask("eps (neighbourhood radius)", 1.0)
min_pts = ask("min_pts (points needed to be core)", 3)

labels = dbscan(points, eps, min_pts)

print(f"\neps = {eps}, min_pts = {min_pts}\n")
print(f"{'Point':>14}   {'Type':<7} Cluster")
for i, (p, lab) in enumerate(zip(points, labels)):
    if len(region_query(points, i, eps)) >= min_pts:           # decide the point type for display
        kind = "Core"
    elif lab != -1:
        kind = "Border"
    else:
        kind = "Noise"
    print(f"{str(p):>14}   {kind:<7} {'-' if lab == -1 else lab}")

n_clusters = len(set(labels) - {-1})                           # distinct labels except noise
print(f"\nNumber of clusters     = {n_clusters}")
print(f"Number of noise points = {labels.count(-1)}")

# ---------- Graphs ----------
SYMBOLS = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"                # symbol for cluster 1, 2, 3, ...
plot_points = [(p[0], p[1], "*" if lab == -1 else SYMBOLS[(lab - 1) % len(SYMBOLS)])
               for p, lab in zip(points, labels)]
scatter_plot(plot_points, title="Clusters: digit = cluster number, * = noise")

if n_clusters:
    names = [f"Cluster {c}" for c in range(1, n_clusters + 1)] + ["Noise"]
    sizes = [labels.count(c) for c in range(1, n_clusters + 1)] + [labels.count(-1)]
    bar_chart(names, sizes, title="Number of points in each cluster")

import numpy as np
import matplotlib.pyplot as plt
import math
import random

# -----------------------------
# Ρυθμίσεις χώρου & RRT
# -----------------------------
X_MIN, X_MAX = 0, 10
Y_MIN, Y_MAX = 0, 10

STEP_SIZE = 0.5          # μήκος βήματος RRT
MAX_ITERS = 3000         # μέγιστος αριθμός επαναλήψεων
GOAL_TOL = 0.5           # πόσο κοντά στον στόχο θεωρούμε ότι φτάσαμε
GOAL_SAMPLE_RATE = 0.05  # πιθανότητα να “δειγματίσουμε” κατευθείαν το goal (goal bias)

START = (1.0, 1.0)
GOAL = (9.0, 9.0)

# Εμπόδια ως ορθογώνια: [xmin, ymin, xmax, ymax]
OBSTACLES = [
    [3.0, 3.0, 7.0, 3.5],
    [3.0, 6.5, 7.0, 7.0],
    [4.5, 3.5, 5.0, 6.5],
]


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def steer(from_node, to_point, step_size):
    """
    Επιστρέφει νέο node που κινείται από το from_node προς το to_point
    με βήμα μήκους step_size (ή λιγότερο αν ο στόχος είναι πολύ κοντά).
    """
    dx = to_point[0] - from_node.x
    dy = to_point[1] - from_node.y
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return Node(from_node.x, from_node.y, parent=from_node)
    scale = min(step_size / d, 1.0)
    new_x = from_node.x + scale * dx
    new_y = from_node.y + scale * dy
    return Node(new_x, new_y, parent=from_node)


def point_in_obstacle(x, y, obstacles):
    for (xmin, ymin, xmax, ymax) in obstacles:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
    return False


def collision_free(node_from, node_to, obstacles, step=0.05):
    """
    Ελέγχει αν το segment from->to τέμνει εμπόδιο.
    Δειγματοληπτεί ενδιάμεσα σημεία.
    """
    dx = node_to.x - node_from.x
    dy = node_to.y - node_from.y
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return not point_in_obstacle(node_from.x, node_from.y, obstacles)
    steps = int(d / step)
    for i in range(steps + 1):
        t = i / max(steps, 1)
        x = node_from.x + t * dx
        y = node_from.y + t * dy
        if point_in_obstacle(x, y, obstacles):
            return False
    return True


def sample_random_point():
    if random.random() < GOAL_SAMPLE_RATE:
        # goal bias
        return (GOAL[0], GOAL[1])
    x = random.uniform(X_MIN, X_MAX)
    y = random.uniform(Y_MIN, Y_MAX)
    return (x, y)


def nearest_node(nodes, point):
    nearest = None
    min_d = float("inf")
    dummy = Node(point[0], point[1])
    for n in nodes:
        d = dist(n, dummy)
        if d < min_d:
            min_d = d
            nearest = n
    return nearest


def reconstruct_path(last_node):
    path = []
    node = last_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    path.reverse()
    return path


def rrt():
    nodes = [Node(START[0], START[1], parent=None)]
    goal_node = Node(GOAL[0], GOAL[1])

    for it in range(MAX_ITERS):
        q_rand = sample_random_point()
        q_near = nearest_node(nodes, q_rand)
        q_new = steer(q_near, q_rand, STEP_SIZE)

        # αγνόησε αν βγει εκτός ορίων
        if not (X_MIN <= q_new.x <= X_MAX and Y_MIN <= q_new.y <= Y_MAX):
            continue

        # collision check
        if not collision_free(q_near, q_new, OBSTACLES):
            continue

        nodes.append(q_new)

        # έλεγχος αν φτάσαμε κοντά στο goal
        if dist(q_new, goal_node) < GOAL_TOL:
            print(f"Goal reached at iteration {it}, nodes: {len(nodes)}")
            return nodes, reconstruct_path(q_new)

    print("Failed to reach goal within max iterations.")
    return nodes, None


def plot_result(nodes, path):
    fig, ax = plt.subplots()
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")

    # εμπόδια
    for (xmin, ymin, xmax, ymax) in OBSTACLES:
        ax.fill(
            [xmin, xmax, xmax, xmin],
            [ymin, ymin, ymax, ymax],
            alpha=0.5,
        )

    # δέντρο RRT
    for n in nodes:
        if n.parent is not None:
            ax.plot([n.x, n.parent.x], [n.y, n.parent.y], linewidth=0.5)

    # start & goal
    ax.plot(START[0], START[1], "go", label="Start")
    ax.plot(GOAL[0], GOAL[1], "ro", label="Goal")

    # τελική διαδρομή
    if path is not None:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=2.5)

    ax.legend()
    ax.set_title("RRT Demo in 2D")
    plt.grid(True)

    # σώσε και ως εικόνα, εκτός από το να εμφανιστεί
    plt.savefig("rrt_result.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    nodes, path = rrt()
    plot_result(nodes, path)

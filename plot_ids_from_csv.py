import csv
import numpy as np
import matplotlib.pyplot as plt


def read_csv(path="ids_replay_log.csv"):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "t": int(row["t"]),
                "attack": int(row["attack"]),
                "d2": float(row["d2"]),
                "thr": float(row["thr"]),
                "ratio": float(row["ratio"]),
                "scale": float(row["scale"]),
                "flagged": int(row["flagged"]),
                "streak": int(row["streak"]),
                "triggered": int(row["triggered"]),
                "alert": int(row["alert"]),
            })
    return rows


def metrics(rows):
    t_attack = None
    for row in rows:
        if row["attack"] == 1:
            t_attack = row["t"]
            break

    t_detect = None
    for row in rows:
        if row["alert"] == 1:
            t_detect = row["t"]
            break

    delay = None
    if t_attack is not None and t_detect is not None:
        delay = t_detect - t_attack

    pre = [r for r in rows if r["attack"] == 0]
    post = [r for r in rows if r["attack"] == 1]

    def rate(arr, key):
        return sum(r[key] for r in arr) / max(1, len(arr))

    out = {
        "t_attack": t_attack,
        "t_detect": t_detect,
        "delay": delay,
        "flag_rate_pre": rate(pre, "flagged"),
        "flag_rate_post": rate(post, "flagged"),
        "trigger_rate_post": rate(post, "triggered"),
        "mean_scale_pre": float(np.mean([r["scale"] for r in pre])) if pre else 0.0,
        "mean_scale_post": float(np.mean([r["scale"] for r in post])) if post else 0.0,
        "max_scale_post": float(np.max([r["scale"] for r in post])) if post else 0.0,
    }
    return out


def main():
    rows = read_csv("ids_replay_log.csv")
    m = metrics(rows)

    t = np.array([r["t"] for r in rows])
    d2 = np.array([r["d2"] for r in rows])
    thr = np.array([r["thr"] for r in rows])
    scale = np.array([r["scale"] for r in rows])
    attack = np.array([r["attack"] for r in rows])

    t_attack = m["t_attack"]

    # Plot 1: d2 vs thr
    plt.figure()
    plt.plot(t, d2, label="Mahalanobis d^2")
    plt.plot(t, thr, label="chi-square threshold")
    if t_attack is not None:
        plt.axvline(t_attack, linestyle="--", label="attack start")
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ids_d2_thr.png", dpi=160)

    # Plot 2: scale
    plt.figure()
    plt.plot(t, scale, label="VO trust scale (R multiplier)")
    if t_attack is not None:
        plt.axvline(t_attack, linestyle="--", label="attack start")
    plt.xlabel("time step")
    plt.ylabel("scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ids_scale.png", dpi=160)

    print("Saved plots: ids_d2_thr.png, ids_scale.png")
    print("Metrics:")
    for k, v in m.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

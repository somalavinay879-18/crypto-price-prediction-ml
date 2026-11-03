"""Draw the accuracy chart used in the README from reports/results.csv.

    python -m crypto_direction.plot
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(results_path: str = "reports/results.csv", out_path: str = "reports/accuracy.png") -> None:
    results = pd.read_csv(results_path).sort_values("accuracy")
    baseline = results.loc[results["model"] == "Majority class", "accuracy"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    colors = ["#9aa5b1" if m == "Majority class" else "#0b6e58" for m in results["model"]]
    bars = ax.barh(results["model"], results["accuracy"] * 100, color=colors, height=0.6)
    ax.axvline(baseline * 100, color="#c97a12", linestyle="--", linewidth=1, label="Majority-class baseline")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    for bar, value in zip(bars, results["accuracy"] * 100):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=9)

    ax.set_xlim(49, 53.5)
    ax.set_xlabel("Accuracy on the last 365 days (%)")
    ax.set_title("Next 15-minute direction, 5 crypto pairs", fontsize=11, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

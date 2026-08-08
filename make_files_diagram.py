"""Render a vertical project-file diagram to figures/fig_files.png."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(6.2, 8.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.axis("off")


def box(x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                linewidth=1.2, edgecolor="#333", facecolor=fc))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=7.4, color="#333")


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=1.3, color="#444"))


box(1.2, 13.2, 7.6, 1.3, "Datasets/",
    "ahsanayub  |  reshabhs-SPML  |  deepset  (CSV corpora)", "#dbe9ff")
box(2.4, 11.3, 5.2, 1.2, "src/pi_pipeline.py",
    "load datasets, Config, metrics, latency", "#e8f5e9")
box(0.8, 9.4, 4.0, 1.2, "src/models.py",
    "TF-IDF  MiniLM+RF/XGB  DistilBERT", "#fff3e0")
box(5.2, 9.4, 4.0, 1.2, "src/plots.py",
    "figure builders (fig1..fig10)", "#fff3e0")
box(0.8, 7.4, 4.0, 1.3, "prompt_injection_\ndetection.ipynb",
    "step-by-step notebook", "#f3e5f5")
box(5.2, 7.4, 4.0, 1.3, "run_experiments.py",
    "-> src/experiment.py", "#f3e5f5")
box(1.2, 5.4, 7.6, 1.3, "Outputs",
    "models/best_model.joblib  figures/  results/*.json,*.csv", "#ede7f6")
box(0.8, 3.4, 4.0, 1.2, "api/app.py",
    "FastAPI /detect (ALLOW/BLOCK)", "#e0f7fa")
box(5.2, 3.4, 4.0, 1.2, "report/report.tex",
    "+ references.bib", "#e0f7fa")

arrow(5.0, 13.2, 5.0, 12.5)
arrow(5.0, 11.3, 5.0, 10.6)
arrow(2.8, 9.4, 2.8, 8.7)
arrow(7.2, 9.4, 7.2, 8.7)
arrow(2.8, 7.4, 3.6, 6.7)
arrow(7.2, 7.4, 6.4, 6.7)
arrow(3.6, 5.4, 2.8, 4.6)
arrow(6.4, 5.4, 7.2, 4.6)

ax.set_title("Project file structure and data flow", fontsize=11, pad=6)
fig.tight_layout()
fig.savefig(os.path.join("figures", "fig_files.png"), dpi=200, bbox_inches="tight")
print("Saved figures/fig_files.png")

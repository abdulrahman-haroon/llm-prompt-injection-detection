"""Generate two report assets:
1. figures/fig0_architecture.png  -- API guardrail deployment diagram.
2. results/error_examples.json     -- false negatives / false positives
   on the unseen deepset set for the best model, for the error analysis.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import joblib

from src.pi_pipeline import FIG_DIR, RES_DIR, load_datasets, Config

# 1. architecture diagram
def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.2, edgecolor="#333", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9, wrap=True)

def arrow(ax, x1, y1, x2, y2, label="", color="#333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.3, color=color))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, label, ha="center",
                fontsize=8, color=color)

fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
box(ax, 0.1, 1.1, 1.5, 0.9, "Client /\nUser prompt", "#dbe9ff")
box(ax, 2.1, 1.1, 2.0, 0.9, "REST API Gateway\n(FastAPI /detect)", "#e8f5e9")
box(ax, 4.6, 1.1, 2.1, 0.9, "ML Screening Layer\n(this project)", "#fff3e0")
box(ax, 7.2, 2.0, 2.6, 0.7, "BLOCK + log\n(injection)", "#ffdede")
box(ax, 7.2, 0.2, 2.6, 0.7, "Forward to LLM\n-> backend / DB", "#e0f7fa")
arrow(ax, 1.6, 1.55, 2.1, 1.55)
arrow(ax, 4.1, 1.55, 4.6, 1.55)
arrow(ax, 6.7, 1.75, 7.2, 2.3, "score>=t", "#c0392b")
arrow(ax, 6.7, 1.35, 7.2, 0.6, "score<t", "#00838f")
ax.set_title("Inline prompt-injection screening in front of an LLM-integrated REST API",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig0_architecture.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("architecture figure saved")

# 2. error examples
cfg = Config()
_, external = load_datasets(cfg)
model = joblib.load(os.path.join("models", "best_model.joblib"))
texts = external["text"].values
y = external["label"].values
scores = model.predict_proba(list(texts))[:, 1]
pred = (scores >= 0.5).astype(int)

fn = [(t, float(s)) for t, s, yy, pp in zip(texts, scores, y, pred) if yy == 1 and pp == 0]
fp = [(t, float(s)) for t, s, yy, pp in zip(texts, scores, y, pred) if yy == 0 and pp == 1]
fn.sort(key=lambda z: z[1])          # most-missed injections
fp.sort(key=lambda z: -z[1])         # most-confident false alarms

def trim(s, n=140):
    s = " ".join(s.split())
    return s[:n] + ("..." if len(s) > n else "")

out = {
    "false_negatives": [{"text": trim(t), "score": round(s, 3)} for t, s in fn[:4]],
    "false_positives": [{"text": trim(t), "score": round(s, 3)} for t, s in fp[:4]],
    "n_fn": len(fn), "n_fp": len(fp),
}
with open(os.path.join(RES_DIR, "error_examples.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))

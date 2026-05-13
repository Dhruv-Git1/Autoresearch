"""Generate heatmap for cross-model anchoring gradient comparison (F5)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
OUT_PATH = os.path.join(OUT_DIR, "cross_model_gradient.png")

# Per-model data: (row_label, family_tag, early%, mid%, late%, late_N_small)
# early = turn <5, mid = 5-14, late = >=15
MODELS = [
    ("R1-7B (primary)",  "R1-Distill (Qwen)",  7.2,  12.8, 28.4, False),
    ("R1-14B",           "R1-Distill (Qwen)",  12.1, 21.1, 31.2, False),
    ("R1-Llama-8B",      "R1-Distill (Llama)",  5.6,  7.3, 41.2, True),
    ("Qwen3-8B",         "Qwen3 (RLVR)",        5.3, 22.0, 66.7, True),
    ("Qwen3-14B",        "Qwen3 (RLVR)",        8.3, 40.8, 75.4, False),
]

FAMILY_COLORS = {
    "R1-Distill (Qwen)":  "#2196F3",
    "R1-Distill (Llama)": "#03A9F4",
    "Qwen3 (RLVR)":       "#FF9800",
}

COL_LABELS = ["Early\n(turns < 5)", "Mid\n(turns 5–14)", "Late\n(turns ≥ 15)"]

# Build data array: shape (n_models, 3)
n_models = len(MODELS)
data = np.array([[m[2], m[3], m[4]] for m in MODELS], dtype=float)

fig, ax = plt.subplots(figsize=(7, 5.5))

# ── Heatmap ───────────────────────────────────────────────────────────────────
cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=80)

im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

# ── Cell annotations ──────────────────────────────────────────────────────────
for i in range(n_models):
    for j in range(3):
        val = data[i, j]
        small_n = (j == 2) and MODELS[i][5]
        txt = f"{val:.1f}%"
        if small_n:
            txt += "*"
        # Choose text colour for contrast
        text_color = "white" if val > 45 else "black"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=10, color=text_color, fontweight="bold")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xticks(range(3))
ax.set_xticklabels(COL_LABELS, fontsize=10.5)
ax.set_yticks(range(n_models))

# Build row labels with family tag and color indicator
row_labels = []
for name, family, *_ in MODELS:
    row_labels.append(f"{name}\n[{family}]")
ax.set_yticklabels(row_labels, fontsize=9)

# Color-code y-tick labels by family
for tick, (name, family, *_) in zip(ax.get_yticklabels(), MODELS):
    tick.set_color(FAMILY_COLORS[family])

# ── Colorbar ─────────────────────────────────────────────────────────────────
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Anchoring rate (%)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# ── Grid lines between cells ─────────────────────────────────────────────────
ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", length=0)

ax.set_title(
    "Length-anchoring gradient across five models and three families\n"
    "(* = late-turn N < 20, treat as indicative; colour = anchoring rate %)",
    fontsize=10.5, pad=10,
)

# ── Family legend ────────────────────────────────────────────────────────────
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=c, label=fam)
    for fam, c in FAMILY_COLORS.items()
]
ax.legend(handles=legend_patches, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5,
          framealpha=0.9, title="Model family", title_fontsize=8.5)

plt.tight_layout(rect=[0, 0.11, 1, 1])
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")

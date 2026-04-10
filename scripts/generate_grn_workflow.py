"""Generate a workflow diagram for TabPFN-based GRN inference pipeline."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(9, 18))
ax.set_xlim(0, 9)
ax.set_ylim(0, 22)
ax.axis("off")

# Colors
C_INPUT = "#E8F5E9"
C_PREPROCESS = "#E3F2FD"
C_TABPFN = "#FFF3E0"
C_EXPR = "#E0F2F1"
C_EDGE = "#F3E5F5"
C_AGGREGATE = "#FCE4EC"
C_OUTPUT = "#C8E6C9"
C_BASELINE = "#F5F5F5"
C_BORDER = "#455A64"
C_ARROW = "#37474F"

PAD = 0.04  # tight box padding


def draw_box(x, y, w, h, title, items, color, title_size=10.5, item_size=9):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={PAD}",
        facecolor=color, edgecolor=C_BORDER, linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h - 0.15, title, ha="center", va="top",
        fontsize=title_size, fontweight="bold", color="#212121",
    )
    for i, item in enumerate(items):
        ax.text(
            x + 0.2, y + h - 0.40 - i * 0.27, f"• {item}",
            ha="left", va="top", fontsize=item_size, color="#424242",
        )


def draw_arrow(x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=2,
                        connectionstyle="arc3,rad=0"),
    )


# ── Title ──
ax.text(4.5, 21.6, "TabPFN for Gene Regulatory Network Inference",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1A237E")

# ── INPUT ──
draw_box(1.75, 20.3, 5.5, 1.0, "INPUT",
         ["Expression matrix (samples × genes) + TF list"],
         C_INPUT)
draw_arrow(4.5, 20.3, 4.5, 19.9)

# ── Step 1: Preprocessing ──
draw_box(1.75, 18.9, 5.5, 1.0, "Step 1: Preprocessing",
         ["Normalize, identify TFs/targets, filter to gold standard"],
         C_PREPROCESS, item_size=8)
draw_arrow(4.5, 18.9, 4.5, 18.5)

# ══════════════════════════════════════════════════════════════
# PHASE A: Expression Prediction
# ══════════════════════════════════════════════════════════════
phase_a = FancyBboxPatch(
    (0.4, 14.6), 8.2, 3.85, boxstyle="round,pad=0.12",
    facecolor="#FAFAFA", edgecolor="#2E7D32", linewidth=2, linestyle="--",
)
ax.add_patch(phase_a)
ax.text(4.5, 18.2, "Phase A: Expression Prediction  (per target gene g)",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2E7D32")

# Feature construction
draw_box(0.7, 16.85, 3.5, 1.1, "Feature Construction",
         ["X = TF expressions, y = target g",
          "Exclude g from X if g ∈ TF set"],
         C_PREPROCESS, item_size=8)

# TabPFN fit
draw_box(4.8, 16.85, 3.8, 1.1, "TabPFN In-Context Learning",
         ["Fit on train fold (no fine-tuning)",
          "Model retained for Phase B"],
         C_TABPFN, item_size=8)
draw_arrow(4.2, 17.4, 4.8, 17.4)

# Expression output
draw_box(1.25, 14.85, 6.5, 1.7, "Expression Prediction Output",
         ["Per-target predictions on held-out test set",
          "Metrics: R², Pearson r, RMSE",
          "TabPFN R²=0.87 (DREAM5) vs 0.70 best baseline"],
         C_EXPR, item_size=8)
draw_arrow(4.5, 16.85, 4.5, 16.55)

# Arrow Phase A → Phase B
draw_arrow(4.5, 14.6, 4.5, 14.25)

# ══════════════════════════════════════════════════════════════
# PHASE B: Edge Prediction
# ══════════════════════════════════════════════════════════════
phase_b = FancyBboxPatch(
    (0.4, 7.0), 8.2, 7.2, boxstyle="round,pad=0.12",
    facecolor="#FAFAFA", edgecolor="#6A1B9A", linewidth=2, linestyle="--",
)
ax.add_patch(phase_b)
ax.text(4.5, 13.95, "Phase B: Edge Prediction  (from fitted models)",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#6A1B9A")

# Strategy label
ax.text(4.5, 13.5, "Extract edge scores (choose strategy):",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#6A1B9A")

# Three strategy boxes
draw_box(0.7, 12.0, 2.4, 1.2, "Integrated Gradients",
         ["Path integration",
          "Best AUROC"],
         C_EDGE, title_size=8.5, item_size=7.5)

draw_box(3.3, 12.0, 2.4, 1.2, "KernelSHAP",
         ["NaN-masking + WLS",
          "Best AUPR"],
         C_EDGE, title_size=8.5, item_size=7.5)

draw_box(5.9, 12.0, 2.4, 1.2, "Attention-Based",
         ["Self-attn / rollout",
          "Fast, degrades at scale"],
         C_EDGE, title_size=8.5, item_size=7.5)

# Arrows from label to boxes
draw_arrow(2.5, 13.4, 1.9, 13.2)
draw_arrow(4.5, 13.4, 4.5, 13.2)
draw_arrow(6.5, 13.4, 7.1, 13.2)

# Per-target edge scores
draw_box(1.5, 10.4, 6, 1.3, "Per-Target Edge Scores",
         ["Score vector: one score per TF for target g",
          "Computed from fitted model (no retraining)"],
         "#FFF8E1", title_size=9.5, item_size=8)
draw_arrow(1.9, 12.0, 3.2, 11.7)
draw_arrow(4.5, 12.0, 4.5, 11.7)
draw_arrow(7.1, 12.0, 5.8, 11.7)

# Multi-fold aggregation
draw_box(1.5, 8.6, 6, 1.5, "Multi-Fold Aggregation",
         ["Average scores across K folds per (TF, target)",
          "3-fold improves AUROC by 3–14%",
          "IG 3-fold only 2.4× slower than 1-fold"],
         C_AGGREGATE, item_size=8)
draw_arrow(4.5, 10.4, 4.5, 10.1)

# Assemble network
draw_box(1.5, 7.2, 6, 1.15, "Network Assembly",
         ["Stack per-target scores → TF × target edge matrix",
          "Rank all (TF, target) pairs by score"],
         C_EXPR, item_size=8)
draw_arrow(4.5, 8.6, 4.5, 8.35)

# Arrow Phase B → Outputs
draw_arrow(4.5, 7.0, 4.5, 6.65)

# ══════════════════════════════════════════════════════════════
# OUTPUTS
# ══════════════════════════════════════════════════════════════
draw_box(0.5, 5.0, 3.7, 1.5, "Output: Expression",
         ["Per-target R², Pearson r",
          "TabPFN R²=0.87 (DREAM5)"],
         C_OUTPUT, item_size=8)

draw_box(4.8, 5.0, 3.7, 1.5, "Output: Edge Predictions",
         ["Ranked regulatory edge list",
          "AUROC, AUPR, Precision@k"],
         C_OUTPUT, item_size=8)

draw_arrow(3, 6.65, 2.35, 6.4)
draw_arrow(6, 6.65, 6.65, 6.4)

# ── Baselines ──
draw_box(0.5, 3.0, 8, 1.7, "Conventional Baselines (for comparison)",
         ["GENIE3: Random Forest importance — fast, strong on small/medium nets",
          "GRNBoost2: Gradient Boosting importance — similar to GENIE3",
          "Mutual Information / Correlation — simplest, competitive"],
         C_BASELINE, title_size=9.5, item_size=8)

plt.tight_layout(pad=0.3)
plt.savefig("results/grn_analysis/tabpfn_grn_workflow.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.savefig("results/grn_analysis/tabpfn_grn_workflow.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: results/grn_analysis/tabpfn_grn_workflow.png and .pdf")

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

CSV_PATH = Path("/workspace/ncm-iwsp-gpt1-finetune-v3/finetune_v3_vs_ncm_gpt1.csv")
OUT = Path("/workspace/ncm-iwsp-gpt1-finetune-v3/finetune_v3_vs_ncm_gpt1_and_gpt1.png")
INIT_STEP = 100000
GPT1_PPL = 18.4
GPT1_CE_NATS = math.log(GPT1_PPL)  # ln(18.4) = 2.91235066461494 if PPL = exp(CE)

rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
v10 = [r for r in rows if r["source"] == "NCM-GPT1 (Kaggle v10)"]
iwsp = [r for r in rows if r["source"] == "NCM-iWSP-GPT1 fine-tune (kernel v3)"]
assert len(v10) == 73 and len(iwsp) == 55
for r in v10 + iwsp:
    for k in ("step", "step_continued", "val_ce_first", "val_ce_mid", "val_ce_last"):
        r[k] = int(float(r[k])) if k in ("step", "step_continued") else float(r[k])

assert v10[0]["step"] == 28000 and v10[-1]["step"] == 100000
assert iwsp[0]["step"] == 0 and iwsp[-1]["step"] == 53232
v10_end = v10[-1]
iw_start = iwsp[0]
iw_end = iwsp[-1]
assert abs(v10_end["val_ce_first"] - 4.594952832907438) < 1e-12
assert abs(iw_start["val_ce_first"] - 4.59708811622113) < 1e-12
assert abs(iw_end["val_ce_last"] - 0.5161383835365996) < 1e-12
assert abs(GPT1_CE_NATS - 2.91235066461494) < 1e-12

COLOR_V10 = "#0072B2"
COLOR_IW = "#D55E00"
COLOR_GPT1 = "#009E73"
METRICS = [
    ("val_ce_first", "val_ce_first"),
    ("val_ce_mid", "val_ce_mid"),
    ("val_ce_last", "val_ce_last"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(13.2, 13.2), dpi=200)
gs = GridSpec(
    3, 3,
    figure=fig,
    height_ratios=[2.35, 0.62, 1.35],
    hspace=0.22,
    wspace=0.20,
    left=0.07,
    right=0.985,
    top=0.855,
    bottom=0.04,
)

fig.suptitle(
    "Validation code CE (NCM) vs official token perplexity (2018 GPT-1)\n"
    "Different metrics and tokenizers — not comparable in magnitude",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)

def style_ax(ax):
    ax.grid(True, which="major", color="#d0d0d0", linewidth=0.6)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.4)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7)

axes = [fig.add_subplot(gs[0, col]) for col in range(3)]
xs_v10 = [r["step_continued"] for r in v10]
xs_iw = [r["step_continued"] for r in iwsp]

for col, (metric, title) in enumerate(METRICS):
    ax = axes[col]
    y_v10 = [r[metric] for r in v10]
    y_iw = [r[metric] for r in iwsp]
    ax.plot(xs_v10, y_v10, color=COLOR_V10, linewidth=1.7, marker="o", markersize=2.4,
            markerfacecolor=COLOR_V10, markeredgewidth=0, zorder=3)
    ax.plot(xs_iw, y_iw, color=COLOR_IW, linewidth=1.9, marker="o", markersize=2.6,
            markerfacecolor=COLOR_IW, markeredgewidth=0, zorder=4)
    ax.scatter([xs_v10[-1]], [v10_end[metric]], s=58, marker="*", color=COLOR_V10,
               edgecolors="black", linewidths=0.4, zorder=6)
    ax.scatter([xs_iw[0]], [iw_start[metric]], s=42, marker="D", facecolors="white",
               edgecolors=COLOR_IW, linewidths=1.2, zorder=6)
    ax.scatter([xs_iw[-1]], [iw_end[metric]], s=42, marker="D", facecolors=COLOR_IW,
               edgecolors="black", linewidths=0.5, zorder=6)
    ax.axvline(INIT_STEP, color="#666666", linestyle="--", linewidth=0.9, zorder=2)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Continued step  (fine-tune at 100,000 + step)")
    if col == 0:
        ax.set_ylabel("Validation code-cell CE (nats)")
    style_ax(ax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: format(int(round(x)), ",")))
    ax.set_xlim(left=min(xs_v10) - 1000)
    if col == 0:
        ax.text(
            -0.02, 1.20,
            "A. NCM validation code-cell cross-entropy  (continuation view, init_step = 100,000)",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="bottom", ha="left", clip_on=False,
        )

handles = [
    Line2D([0], [0], color=COLOR_V10, linewidth=1.8, marker="o", markersize=4, label="NCM-GPT1 (Kaggle v10)"),
    Line2D([0], [0], color=COLOR_IW, linewidth=1.8, marker="o", markersize=4, label="NCM-iWSP-GPT1 fine-tune (kernel v3)"),
    Line2D([0], [0], marker="*", color=COLOR_V10, markeredgecolor="black", markeredgewidth=0.4,
           linestyle="None", markersize=10, label="v10 at 100,000"),
    Line2D([0], [0], marker="D", markerfacecolor="white", markeredgecolor=COLOR_IW, markeredgewidth=1.2,
           linestyle="None", markersize=7, label="Fine-tune start (converted init, step 0)"),
    Line2D([0], [0], marker="D", markerfacecolor=COLOR_IW, markeredgecolor="black", markeredgewidth=0.5,
           linestyle="None", markersize=7, label="Fine-tune end (step 53,232)"),
    Line2D([0], [0], color="#666666", linestyle="--", linewidth=0.9, label="Join at init_step = 100,000"),
]
fig.legend(
    handles=handles, loc="upper center", ncol=2, frameon=False,
    bbox_to_anchor=(0.5, 0.922), columnspacing=1.3, handletextpad=0.5,
)

# Reserved middle row for the NCM annotation (no overlap with axes)
ax_note = fig.add_subplot(gs[1, :])
ax_note.axis("off")
ncm_note = (
    "NCM-GPT1 v10 used alpha=0.5, mid_aux=0.5; the iWSP fine-tune used alpha=0, mid_aux=0.\n"
    "Only val_ce_first, val_ce_mid, and val_ce_last are comparable between these two NCM runs — not the optimized total loss.\n"
    "v10 val: 73 points from step 28,000 to 100,000 (256 windows). Fine-tune val: 55 points from step 0 to 53,232 (finished=false; stopped by the 11.5 h / 15 min reserve limit).\n"
    "The discontinuity at the join is mostly from conversion, not from later training: the converted iWSP init (untied heads, no post-upsample conv) already differs from v10 at 100k on mid/last; first is nearly unchanged."
)
ax_note.text(
    0.0, 0.72, ncm_note,
    ha="left", va="top", fontsize=8.2, color="#222222",
    transform=ax_note.transAxes, linespacing=1.38, wrap=True,
)

# Bottom: official 2018 GPT-1 single number, separate units
gs_b = gs[2, :].subgridspec(1, 2, width_ratios=[1.05, 2.15], wspace=0.10)
ax_bar = fig.add_subplot(gs_b[0, 0])
ax_txt = fig.add_subplot(gs_b[0, 1])

ax_bar.bar([0], [GPT1_PPL], width=0.55, color=COLOR_GPT1, edgecolor="black", linewidth=0.6, zorder=3)
ax_bar.scatter([0], [GPT1_PPL], s=70, marker="o", color=COLOR_GPT1, edgecolors="black", linewidths=0.6, zorder=4)
ax_bar.set_xlim(-0.9, 0.9)
ax_bar.set_ylim(0, 26)
ax_bar.set_xticks([0])
ax_bar.set_xticklabels(["2018 GPT-1\n(BooksCorpus)"])
ax_bar.set_ylabel("Official token-level perplexity")
ax_bar.set_title("Official published number (one point)", fontweight="bold", fontsize=11)
style_ax(ax_bar)
ax_bar.grid(False, which="minor")
ax_bar.annotate(
    f"{GPT1_PPL:g}\n≡ {GPT1_CE_NATS:.3f} nats\n(if PPL = exp(CE))",
    xy=(0, GPT1_PPL),
    xytext=(0, 21.0),
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
    color="#004C3A",
)

ax_txt.axis("off")
ax_txt.set_xlim(0, 1)
ax_txt.set_ylim(0, 1)
# Light background to separate this panel visually from the NCM CE row
bbox = FancyBboxPatch(
    (0.0, 0.02), 1.0, 0.96,
    boxstyle="round,pad=0.01,rounding_size=0.02",
    transform=ax_txt.transAxes,
    facecolor="#f6f7f5",
    edgecolor="#cccccc",
    linewidth=0.6,
    clip_on=False,
    zorder=0,
)
ax_txt.add_patch(bbox)
ax_txt.text(
    0.03, 0.93,
    "B. Official 2018 GPT-1 token-level result  —  no training curve was published",
    fontsize=11.5, fontweight="bold", va="top", ha="left", transform=ax_txt.transAxes,
    clip_on=False, zorder=1,
)
ax_txt.text(
    0.03, 0.78,
    "Paper: Radford et al., Improving Language Understanding by Generative Pre-Training (2018), §4.1.\n"
    "Quote: “Our language model achieves a very low token level perplexity of 18.4 on this corpus.”\n"
    "The paper does not say whether 18.4 is training or validation perplexity, and it publishes no curve.\n"
    "Setup stated in §4.1: 12-layer decoder-only Transformer, 768-d, 12 heads, context 512, BooksCorpus,\n"
    "BPE with 40,000 merges, 100 epochs, minibatches of 64 sequences of 512 tokens, Adam max lr 2.5e-4.\n"
    "The 2018 paper does not state a parameter count.\n"
    "Equivalent next-token CE if PPL = exp(CE) with natural log: ln(18.4) = 2.912 nats.\n\n"
    "This panel is a single official reference, not a training curve. The metric is next-token PPL\n"
    "on BooksCorpus, with a 40,000-merge BPE tokenizer (not GPT-2 BPE). It is NOT comparable in\n"
    "magnitude to the NCM code-cell CE panels above, and is not plotted on those axes.",
    fontsize=8.2, va="top", ha="left", transform=ax_txt.transAxes, linespacing=1.38, color="#222222",
    zorder=1,
)

fig.savefig(OUT, dpi=200, facecolor="white", edgecolor="none")
print("wrote", OUT, "bytes", OUT.stat().st_size)
prev = Path("/workspace/ncm-iwsp-gpt1-finetune-v3/finetune_v3_vs_ncm_gpt1.png")
print("previous_png_unchanged_bytes", prev.stat().st_size, "mtime", prev.stat().st_mtime)
print("gpt1_ppl", GPT1_PPL, "ln", GPT1_CE_NATS)

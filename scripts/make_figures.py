"""Generate publication-quality figures for the COMPASS paper.

Figure A: sensitivity.pdf  (double-column, 3 panels: tau / alpha / k)
Figure B: ablation.pdf      (single-column, vertical bar chart, blue gradient)

All numbers are taken verbatim from the prompt / docs/paper_assets and are NOT
altered. AVG-HR: lower is better.

Visual style: a clean blue-gradient palette (deep blue -> pale light blue),
minimal chartjunk, thin spines, tasteful top-conference look.

Also writes 150-dpi PNG previews to /tmp for visual inspection.
"""
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Refined, restrained aesthetic
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# ---- blue-gradient palette (reference style) ----
BLUE_DEEP  = "#1F5FBF"   # strongest deep blue (best)
BLUE       = "#2C6FBB"   # primary blue (sensitivity lines/markers)
BLUE_PALE  = "#AEC7E8"   # pale light blue (worst end of gradient)
ACCENT     = "#E08A3C"   # warm orange accent (paper default star)
INK        = "#2B2B2B"   # near-black for text
GRID       = "#E6E6E6"   # faint gridlines
BANDC      = "#DCE6F2"   # light blue-gray robustness band

OUT = "paper/figures"


def _save(fig, pdf_path, png_path):
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)


def _lerp_hex(c0, c1, t):
    """Linear interpolation between two hex colors; t in [0,1]."""
    def to_rgb(c):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    a, b = to_rgb(c0), to_rgb(c1)
    rgb = tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))
    return "#{:02X}{:02X}{:02X}".format(*rgb)


# ===========================================================================
# Figure A — Sensitivity (3 side-by-side panels, shared y) — blue family
# ===========================================================================
tau_x   = [0.40, 0.45, 0.50, 0.55, 0.60]
tau_y   = [37.8, 38.0, 36.0, 35.8, 37.0]
alpha_x = [0.2, 0.3, 0.4, 0.5, 0.6]
alpha_y = [38.2, 37.5, 35.8, 37.2, 35.8]
k_x     = [3, 5, 7]
k_y     = [37.0, 35.8, 35.8]

panels = [
    (r"$\tau$  ", "gate threshold", tau_x,   tau_y,   0.55, "{:.2f}"),
    (r"$\alpha$  ", "gate weight",   alpha_x, alpha_y, 0.40, "{:.1f}"),
    (r"$k$  ", "TRACE samples",      k_x,     k_y,     5,    "{:d}"),
]

# shared y-limits with gentle headroom
all_y = tau_y + alpha_y + k_y
ymin, ymax = min(all_y), max(all_y)
ylo, yhi = ymin - 0.9, ymax + 1.3

# soft robustness band: a 2-point window above the global optimum
band_lo = ymin
band_hi = ymin + 2.0

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
fig.subplots_adjust(wspace=0.12)

for ax, (sym, desc, xs, ys, default_x, xfmt) in zip(axes, panels):
    # soft shaded robustness band (global, so it lines up across panels)
    ax.axhspan(band_lo, band_hi, color=BANDC, zorder=0, linewidth=0)

    # light baseline at the global optimum
    ax.axhline(ymin, color=BLUE, lw=0.7, ls=(0, (4, 3)), alpha=0.45, zorder=1)

    # faint horizontal gridlines only
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # thin connecting line with refined open markers, deep blue
    ax.plot(xs, ys, color=BLUE, lw=1.3, zorder=3,
            marker="o", markersize=4.2, markerfacecolor="white",
            markeredgecolor=BLUE, markeredgewidth=1.2)

    # default point: filled star in the contrasting warm accent
    di = xs.index(default_x)
    ax.plot(default_x, ys[di], marker="*", markersize=13,
            color=ACCENT, markeredgecolor="white", markeredgewidth=0.5,
            zorder=6, linestyle="None")
    ax.annotate(
        "default",
        xy=(default_x, ys[di]), xytext=(0, 11), textcoords="offset points",
        ha="center", va="bottom", fontsize=6.8, color=ACCENT,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.6,
                        shrinkA=0.0, shrinkB=5.0, alpha=0.7),
    )

    # title: bold symbol + light descriptor
    ax.set_title(desc, fontsize=8, color="0.45", pad=12)
    ax.text(0.5, 1.18, sym, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, color=INK, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([xfmt.format(x) for x in xs])
    ax.set_ylim(ylo, yhi)
    xr = (max(xs) - min(xs)) or 1
    ax.set_xlim(min(xs) - 0.10 * xr, max(xs) + 0.10 * xr)
    ax.tick_params(length=2.5, width=0.8, colors="0.3")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("0.4")

axes[0].set_ylabel("AVG-HR  (lower is better)", fontsize=8.5)

legend_elems = [
    Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white",
           markeredgecolor=BLUE, markersize=5, lw=1.3, label="swept value"),
    Line2D([0], [0], color=ACCENT, marker="*", markersize=11,
           linestyle="None", label="paper default"),
    Patch(facecolor=BANDC, edgecolor="none", label="robustness band ($<\\!2$ pt)"),
]
fig.legend(handles=legend_elems, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.10), handletextpad=0.5, columnspacing=1.6)

fig.tight_layout(rect=(0, 0.0, 1, 0.90))
_save(fig, f"{OUT}/sensitivity.pdf", "/tmp/sens_v2.png")
plt.close(fig)

# ===========================================================================
# Figure B — Ablation (vertical bar chart, blue gradient, zoomed honest axis)
# ===========================================================================
# Ordered best (left) -> worst (right) so the eye reads degradation rightward.
abl = [
    ("Full\nCOMPASS", 36.3),
    (r"$-$TRACE",     37.4),
    (r"$-$CPC",       38.2),
    (r"$-$both",      39.3),
]
adaptive_rag = 39.7

labels = [a[0] for a in abl]
vals   = [a[1] for a in abl]
xpos   = list(range(len(vals)))

# Gradient by value: best (Full COMPASS) deepest blue -> worst palest blue.
n = len(vals)
bar_colors = [_lerp_hex(BLUE_DEEP, BLUE_PALE, i / (n - 1)) for i in range(n)]

fig2, ax2 = plt.subplots(figsize=(3.4, 2.6))

# Zoomed (honest) y-axis: values lie in 36-39.7, so start near 35.
ylo2, yhi2 = 35.0, 40.5

# faint horizontal gridlines only
ax2.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax2.set_axisbelow(True)

bars = ax2.bar(xpos, vals, width=0.64, color=bar_colors,
               edgecolor="white", linewidth=0.6, zorder=3)

# value label above each bar
for x, v in zip(xpos, vals):
    ax2.text(x, v + 0.10, f"{v:.1f}", ha="center", va="bottom",
             fontsize=8, color=INK, fontweight="bold", zorder=5)

# Adaptive-RAG reference: horizontal dashed line + label
ax2.axhline(adaptive_rag, color="0.55", linestyle=(0, (4, 3)), linewidth=1.0,
            zorder=2)
ax2.text(len(vals) - 0.45, adaptive_rag + 0.06,
         f"Adaptive-RAG  {adaptive_rag:.1f}",
         ha="right", va="bottom", fontsize=6.8, color="0.45", style="italic")

ax2.set_ylim(ylo2, yhi2)
ax2.set_xlim(-0.65, len(vals) - 0.35)
ax2.set_xticks(xpos)
ax2.set_xticklabels(labels, fontsize=8)

# bold the Full COMPASS tick label, in the deep blue
for tick, lab in zip(ax2.get_xticklabels(), labels):
    if lab.startswith("Full"):
        tick.set_fontweight("bold")
        tick.set_color(BLUE_DEEP)

ax2.set_ylabel("AVG-HR  (lower is better)", fontsize=8.5)
ax2.set_yticks([35, 36, 37, 38, 39, 40])
ax2.tick_params(length=2.5, width=0.8, colors="0.3")
ax2.tick_params(axis="x", length=0)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax2.spines[sp].set_color("0.4")

fig2.tight_layout()
_save(fig2, f"{OUT}/ablation.pdf", "/tmp/abl_v2.png")
plt.close(fig2)


# ===========================================================================
# Figure C — Main results headline bar chart (AVG-HR across 4 benchmarks)
# ===========================================================================
# AVG Hallucination Rate (%, LOWER is better), verbatim from
# docs/paper_assets/01_main_table.md. Sorted ascending (best -> worst).
# Honesty: COMPASS (36.3) and Rowen (36.4) are essentially tied; the chart
# starts at y=0 so the near-equal heights are read truthfully, with the
# sorted order conveying "on par with the best baseline, well ahead of rest."
def make_mainbar():
    BLUE_MED = "#5B8FD4"   # medium blue for the runner-up (Rowen)

    methods = [
        ("COMPASS\n(Ours)", 36.3, "ours"),
        ("Rowen",           36.4, "runner"),
        ("Adaptive-RAG",    39.7, "other"),
        ("Direct",          41.2, "other"),
        ("Self-Refine",     41.4, "other"),
        ("CoK",             43.1, "other"),
        ("HaluSearch",      47.4, "other"),
        ("ReAct",           49.5, "other"),
    ]

    labels = [m[0] for m in methods]
    vals   = [m[1] for m in methods]
    kinds  = [m[2] for m in methods]
    xpos   = list(range(len(vals)))

    colors = []
    for k in kinds:
        if k == "ours":
            colors.append(BLUE_DEEP)
        elif k == "runner":
            colors.append(BLUE_MED)
        else:
            colors.append(BLUE_PALE)

    fig3, ax3 = plt.subplots(figsize=(3.4, 2.7))

    # honest axis from 0; headroom above tallest bar for value labels
    ylo3, yhi3 = 0.0, 54.0

    ax3.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax3.set_axisbelow(True)

    bars = ax3.bar(xpos, vals, width=0.70, color=colors,
                   edgecolor="white", linewidth=0.6, zorder=3)

    # value labels on every bar; bold for COMPASS
    for x, v, k in zip(xpos, vals, kinds):
        ax3.text(x, v + 0.7, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=7.5 if k != "ours" else 8.2,
                 color=BLUE_DEEP if k == "ours" else INK,
                 fontweight="bold" if k == "ours" else "normal",
                 zorder=5)

    # "Ours" callout: small downward arrow above the COMPASS bar
    ax3.annotate(
        "Ours", xy=(0, vals[0] + 4.2), xytext=(0, vals[0] + 12.5),
        ha="center", va="bottom", fontsize=8.5, color=BLUE_DEEP,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=BLUE_DEEP, lw=1.3,
                        shrinkA=1.0, shrinkB=2.0),
        zorder=6,
    )

    ax3.set_ylim(ylo3, yhi3)
    ax3.set_xlim(-0.65, len(vals) - 0.35)
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(labels, fontsize=7, rotation=30, ha="right",
                        rotation_mode="anchor")

    # bold + deep-blue the COMPASS tick label
    for tick, lab in zip(ax3.get_xticklabels(), labels):
        if lab.startswith("COMPASS"):
            tick.set_fontweight("bold")
            tick.set_color(BLUE_DEEP)

    ax3.set_ylabel("Avg. Hallucination Rate (%, lower is better)",
                   fontsize=8.0)
    ax3.set_yticks([0, 10, 20, 30, 40, 50])
    ax3.tick_params(length=2.5, width=0.8, colors="0.3")
    ax3.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax3.spines[sp].set_color("0.4")

    fig3.tight_layout()
    _save(fig3, f"{OUT}/mainbar.pdf", "/tmp/mainbar.png")
    plt.close(fig3)


make_mainbar()

print("wrote sensitivity.pdf + ablation.pdf + mainbar.pdf (and /tmp PNG previews)")

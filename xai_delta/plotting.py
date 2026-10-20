"""Figure style of the paper and the plots used in the notebooks."""
import shutil
import subprocess

import matplotlib as mpl
import numpy as np


def _latex_available():
    """LaTeX text rendering in matplotlib needs `latex` and the cm-super fonts (type1ec.sty)."""
    if shutil.which("latex") is None or shutil.which("kpsewhich") is None:
        return False
    return subprocess.run(["kpsewhich", "type1ec.sty"], capture_output=True, text=True).stdout.strip() != ""


USE_TEX = _latex_available()

_BASE = {
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
_TEX = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{bm}",
}
# Without LaTeX: the Computer Modern bold font shipped with matplotlib (all text is bold in the
# paper figures) and Computer Modern for math.
_NO_TEX = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmb10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": False,
    "axes.unicode_minus": False,
}

PRE = r"$M_{\mathrm{pre}}$"
POST = r"$M_{\mathrm{post}}$"


def use_paper_style():
    """Apply the rcParams used for the figures of the paper."""
    mpl.rcParams.update({**_BASE, **(_TEX if USE_TEX else _NO_TEX)})


def bold(text):
    """Bold text for tick and axis labels (LaTeX special characters escaped)."""
    if not USE_TEX:
        return str(text)
    escaped = str(text)
    for char in "%&#_":
        escaped = escaped.replace(char, "\\" + char)
    return rf"\textbf{{{escaped}}}"


def _finish(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(True, linewidth=0.6, alpha=0.3)


def plot_layer_similarity(ax, mean, sem, highlight_block, ylabel):
    """Layer-wise similarity between M_pre and M_post (mean with SEM band over prompts),
    with the analysis layer circled. `mean` and `sem` exclude the embedding layer."""
    blocks = np.arange(1, len(mean) + 1)
    ax.plot(blocks, mean, marker="o", markersize=4, color="black")
    ax.fill_between(blocks, mean - sem, mean + sem, alpha=0.2)
    ax.plot(highlight_block, mean[highlight_block - 1], marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor="red", markeredgewidth=2.2, linestyle="None", zorder=5)

    step = max(1, len(blocks) // 6 // 4 * 4)
    xticks = list(range(step, len(blocks) + 1, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([bold(t) for t in xticks])
    low = (mean - sem).min()
    lo = np.floor(low * 10) / 10 if low < 0.9 else np.floor(low * 100) / 100
    ax.set_ylim(lo, 1 + 0.05 * (1 - lo))
    yticks = [round(t, 3) for t in np.linspace(lo, 1.0, 3)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([bold(f"{t:g}") for t in yticks])

    ax.set_xlabel(bold("Transformer layer"), fontsize=20)
    ax.set_ylabel(rf"\textbf{{\shortstack[c]{{{ylabel}\\similarity}}}}" if USE_TEX else f"{ylabel}\nsimilarity",
                  fontsize=20, labelpad=12)
    _finish(ax)


def plot_patching_boxplot(ax, sim_post, sim_patched):
    """Cosine similarity to M_pre's output of the unpatched and patched M_post outputs
    (box plots with the mean as a triangle)."""
    bp = ax.boxplot([sim_post, sim_patched], patch_artist=True, showmeans=True, meanline=False, widths=0.55)
    for patch, color in zip(bp["boxes"], ["firebrick", "slateblue"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.6)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set_color("black")
            artist.set_linewidth(1.6)
    for flier in bp["fliers"]:
        flier.set(marker="o", markersize=3, markerfacecolor="black", markeredgecolor="black", alpha=0.6)
    for mean in bp["means"]:
        mean.set(marker="^", markersize=7, markerfacecolor="white", markeredgecolor="black",
                 markeredgewidth=1.6, linestyle="None")

    ax.set_ylabel(bold("Cosine similarity"), fontsize=20, labelpad=10)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([bold("Post vs Pre"), bold("Patched vs Pre")])
    lo, hi = ax.get_ylim()
    yticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    ax.set_yticks(yticks)
    ax.set_yticklabels([bold(f"{t:.2f}") for t in yticks])
    ax.set_ylim(lo, hi)
    _finish(ax)


def plot_concept_shifts(ax, labels, deltas):
    """Horizontal bars of the mean activation shift of the most changed SAE features."""
    order = np.argsort(deltas)
    y = np.arange(len(order))
    ax.barh(y, np.asarray(deltas)[order], color="black", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i].replace("&", r"\&") if USE_TEX else labels[i] for i in order], fontsize=12)
    ax.set_title(bold("Most changed SAE features"), fontsize=13)
    ax.set_xlabel(bold("Average feature activation difference") + rf" ({POST} $-$ {PRE})", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)


def plot_grouped_bars(ax, table, ylabel, ylim=None):
    """Grouped bars (one group per row of `table`, one bar per column) in the paper's colors."""
    colors = ["slateblue", "firebrick", "seagreen"]
    n_rows, n_cols = table.shape
    width = 0.8 / n_cols
    x = np.arange(n_rows)
    for j, col in enumerate(table.columns):
        ax.bar(x + (j - (n_cols - 1) / 2) * width, table[col], width, label=col, color=colors[j % 3],
               alpha=0.8, edgecolor="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([bold(i) for i in table.index], fontsize=14)
    ax.set_ylabel(bold(ylabel), fontsize=18)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=12, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n_cols)
    _finish(ax)
    ax.grid(False, axis="x")

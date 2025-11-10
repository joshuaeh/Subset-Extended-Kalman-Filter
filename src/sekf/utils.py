import numpy as np
from scipy import stats
from pathlib import Path
import zipfile


def get_ci(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf(
        (1 + confidence) / 2, n - 1
    )  # get the t-score corresponding to the confidence interval
    return m, h


def plot_ci(
    data_x,
    data_y,
    ax,
    label=None,
    color=None,
    marker_fmt="o",
    marker_edgecolor="black",
    marker_edgewidth=0.5,
    marker_size=50,
    barx=True,
    bary=True,
    bar_color="black",
    bar_capsize=5,
    bar_thickness=1,
):
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]

    x, ex = get_ci(data_x)
    y, ey = get_ci(data_y)

    ex = ex if barx else None
    ey = ey if bary else None

    ax.errorbar(
        x,
        y,
        xerr=ex,
        yerr=ey,
        label=label,
        fmt="none",
        ecolor=bar_color,
        capsize=bar_capsize,
        linewidth=bar_thickness,
        zorder=1,
    )

    return ax.scatter(
        x,
        y,
        marker=marker_fmt,
        facecolor=color,
        edgecolor=marker_edgecolor,
        linewidth=marker_edgewidth,
        s=marker_size,
        zorder=2,
    )

def zip_dir(dir, zip_filename):
    dir = Path(dir)
    for file in dir.rglob("*"):
        if file.is_file():
            with zipfile.ZipFile(zip_filename, "a") as zipf:
                zipf.write(file, file.relative_to(dir))
    return

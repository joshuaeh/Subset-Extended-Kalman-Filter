"""Configuration"""
# imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# plotting parameters
# plot style
# sns.set_style("whitegrid")
color_palette = sns.color_palette("colorblind")
sns.set_palette(color_palette)
# sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_style({"xtick.bottom": True, "ytick.left": True})

# plt.rc("font", family="Arial")
# plt.rc("font", family="sans-serif", size=12)
# plt.rc("axes", labelsize=7)
# plt.rc("legend", fontsize=7)
# plt.rc("xtick", labelsize=5)
# plt.rc("ytick", labelsize=5)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 5
plt.rcParams["axes.titlesize"] = 7
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["xtick.labelsize"] = 5
plt.rcParams["ytick.labelsize"] = 5
# Set font as TrueType
plt.rcParams["pdf.fonttype"] = 42

plt.rc("savefig", dpi=1_000, bbox="tight", pad_inches=0.01)

# colors
colors = sns.color_palette("tab20c")
model_colors = colors[:4]
emph_colors = colors[4:8]
kalman_colors = colors[8:12]
adam_colors = colors[12:16]
true_colors = colors[16:]
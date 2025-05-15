import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# apply settings
rng = np.random.default_rng(seed=42)
# torch.set_default_dtype(torch.float32)
torch_seed = torch.manual_seed(42)
device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     torch.set_default_device(device)
#     # x = torch.ones(1)
#     # print(x)
# elif torch.backends.mps.is_available():
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#     device = torch.device("cpu")
#     torch.set_default_device(device)
    
#     # device = torch.device("mps")
#     # torch.set_default_device(device)
    
#     # x = torch.ones(1)
#     # print(x)
# else:
#     device = torch.device("cpu")
#     torch.set_default_device(device)
#     print("Cuda or MPS device not found.")

# plot style
# sns.set_style("whitegrid")
color_palette = sns.color_palette("colorblind")
sns.set_palette(color_palette)
# sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_style({"xtick.bottom": True, "ytick.left": True})
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["figure.dpi"] = 1000

# plt.rc("font", family="Arial")
# plt.rc("font", family="sans-serif", size=12)
# plt.rc("axes", labelsize=7)
# plt.rc("legend", fontsize=7)
# plt.rc("xtick", labelsize=5)
# plt.rc("ytick", labelsize=5)
plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 7
plt.rcParams["axes.titlesize"] = 7
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
# Set font as TrueType
plt.rcParams["pdf.fonttype"] = 42

# plt.rc("savefig", dpi=1_000, bbox="tight", pad_inches=0.01)
plt.rc("savefig", dpi=1_000)
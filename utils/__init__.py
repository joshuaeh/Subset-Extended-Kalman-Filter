# TODO: refactor to config.py
# from config import *

# imports
import datetime
import logging
import os
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import scipy.stats as stats
import seaborn as sns
import h5py
import torch

# import other sister modules

# apply settings
default_rng = np.random.default_rng(seed=42)
torch.set_default_dtype(torch.float32)
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

# plt.rc("font", family="Arial")
# plt.rc("font", family="sans-serif", size=12)
# plt.rc("axes", labelsize=7)
# plt.rc("legend", fontsize=7)
# plt.rc("xtick", labelsize=5)
# plt.rc("ytick", labelsize=5)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
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

# constants
bytes_dict = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}

colors = sns.color_palette("tab20c")
model_colors = colors[:4]
emph_colors = colors[4:8]
kalman_colors = colors[8:12]
adam_colors = colors[12:16]
true_colors = colors[16:]

def get_ci(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n-1)  # get the t-score corresponding to the confidence interval
    return m, h

def plot_ci(data_x, data_y, ax, label=None,
    color=None, marker_fmt="o", marker_edgecolor="black", marker_edgewidth=0.5, marker_size=50,
    barx=True, bary=True, bar_color="black", bar_capsize=5, bar_thickness=1):
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    
    x, ex = get_ci(data_x)
    y, ey = get_ci(data_y)
    
    ex = ex if barx else None
    ey = ey if bary else None 
    
    ax.errorbar(x, y, xerr=ex, yerr=ey, label=label, fmt="none", ecolor=bar_color, capsize=bar_capsize, linewidth=bar_thickness, zorder=1)
    
    return ax.scatter(x, y, marker=marker_fmt, facecolor=color, edgecolor=marker_edgecolor, linewidth=marker_edgewidth, 
        s=marker_size, zorder=2)

def format_ci_dict(loss_values, time_values, name):
    ci_dict = {}
    loss_mean, loss_ci = get_ci(loss_values)
    time_mean, time_ci = get_ci(time_values)
    ci_dict["Name"] = name
    ci_dict["Loss (Mean)"] = loss_mean
    ci_dict["Loss 95% CI +/-"] = loss_ci[0]
    ci_dict["Time (Mean)"] = time_mean
    ci_dict["Time 95% CI +/-"] = time_ci[0]
    return ci_dict

def table_legend(handles_array, row_labels, column_labels):
    """" 
    ---
    ARGS:
    ---
    handles_array: np.array of handles in the shape (row and columns) that is finally desired
    row_labels: list of row labels
        len: handles_array.shape[0] + 1 or handles_array.shape[0]
        ndim: 1
    column_labels: list of column labels
        len: handles_array.shape[1] 
        ndim: 1
        
    RETURNS:
    ---
    legend_handle: np.array of handles in the shape (row and columns) that is finally desired
    legend_labels: np.array of labels in the shape (row and columns) that is finally desired
    n_col: int, number of columns in the legend
    """
    # shape conditions
    handles_array_shape = handles_array.shape
    rows_labels_matches_handles = handles_array_shape[0] == len(row_labels)
    extra_row_label = handles_array_shape[0] + 1 == len(row_labels)
    columns_labels_matches_handles = handles_array_shape[1] == len(column_labels)
    extra_column_label = handles_array_shape[1] + 1 == len(column_labels)
    assert (rows_labels_matches_handles or extra_row_label), f"Row labels should be handles_array.shape[0] or handles_array.shape[0].\nRecieved: {len(row_labels)=} and {handles_array_shape[0] + 1=}"
    assert (columns_labels_matches_handles or extra_column_label), f"Column labels should be handles_array.shape[1].\nRecieved: {len(column_labels)=} and {handles_array_shape[1]=}"
    
    # format handles
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    ## create new holder array that has empty handle
    _handles_array = np.empty((handles_array_shape[0] + 1, handles_array_shape[1] + 1), dtype=object)
    ## prepend empty handle in first row and column
    _handles_array[0,:] = extra
    _handles_array[:,0] = extra
    # fill in handles
    _handles_array[1:,1:] = handles_array
    ## Transpose (since columns fill by column) and flatten
    legend_handle = _handles_array.T.flatten()
    
    # legend_labels
    empty_label = [""]
    assert np.array(row_labels).ndim == 1, f"Row labels should be 1D array.\nRecieved: {row_labels.ndim=}"
    assert np.array(column_labels).ndim == 1, f"Column labels should be 1D array.\nRecieved: {column_labels.ndim=}"
    if not extra_row_label:
        if extra_column_label:
            row_labels = np.concatenate([[column_labels[0]], row_labels])
            column_labels = column_labels[1:]
        else:
            # add empty label
            row_labels = np.concatenate([empty_label, row_labels])
    _legend_labels = [row_labels]
    for col_label in column_labels:
        _legend_labels.append([col_label])
        _legend_labels.append(empty_label * handles_array_shape[0])
    legend_labels = np.concatenate(_legend_labels)
    
    return legend_handle, legend_labels, handles_array_shape[1]+1

# functions
def numpy_memory_size(numpy_array, units="MB"):
    """Get the memory size of a numpy array"""
    return numpy_array.nbytes / bytes_dict[units]

class h5_logger:
    """Class to log data to an hdf5 file. The data is stored in datasets with the key being the name of the dataset."""
    # TODO: group logger?
    # TODO: removing key
    # TODO: assert that data values are numpy arrays
    def __init__(self, filename, replace=False):
        self.filename = filename
        existing = os.path.exists(filename)
        if existing:
            if replace:
                os.remove(filename)
            else:
                logging.debug(f"File {filename} already exists. Use replace=True to overwrite.")
        else:
            with h5py.File(filename, "w") as file:
                file.attrs["created"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _maxshape(self, data):
        return (None,) + data.shape

    def _init_dataset(self, file, dataset_name, data):
        try:
            file.create_dataset(dataset_name, data=data[None], maxshape=self._maxshape(data))
        except BlockingIOError:
            logging.error("BlockingIOError: Retrying")
            time.sleep(1)
            file.create_dataset(dataset_name, data=data[None], maxshape=self._maxshape(data))

    def _append_to_dataset(self, file, dataset_name, data):
        try:
            file[dataset_name].resize((file[dataset_name].shape[0] + 1, *file[dataset_name].shape[1:]))
            file[dataset_name][-1] = data
        except BlockingIOError:
            logging.error("BlockingIOError: Retrying")
            time.sleep(1)
            file[dataset_name].resize((file[dataset_name].shape[0] + 1, *file[dataset_name].shape[1:]))
            file[dataset_name][-1] = data
            
    def _del_dataset(self, file, dataset_name):
        del file[dataset_name]
        
    def recursive_del(self, key):
        with h5py.File(self.filename, "r+") as file:
            for k in file[key].keys():
                if isinstance(file[key][k], h5py.Group):
                    self.recursive_del(f"{key}/{k}")
                else:
                    del file[key][k]
            del file[key]

    def log_attribute(self, key, value, replace=False):
        """Does not add an extra dimension, designed to be set once."""
        if self.check_key(key):
            if not replace:
                AttributeError(f"Key {key} already exists. Use replace=True to overwrite.")
            else:
                with h5py.File(self.filename, "a") as file:
                    del file[key]
                    file[key] = value
        else:        
            with h5py.File(self.filename, "a") as file:
                file[key] = value
        
    
    def log_value(self, data_key, data_value, file=None):
        if file is not None:
            if data_key not in file.keys():
                self._init_dataset(file, data_key, data_value)
            else:
                self._append_to_dataset(file, data_key, data_value)
        else:
            with h5py.File(self.filename, "a") as file:
                if data_key not in file.keys():
                    self._init_dataset(file, data_key, data_value)
                else:
                    self._append_to_dataset(file, data_key, data_value)

    def log_dict(self, data_dict):
        with h5py.File(self.filename, "a") as file:
            for key, value in data_dict.items():
                self.log_value(key, value, file=file)        

    def open_log(self):
        return h5py.File(self.filename, "a")

    def get_dataset(self, dataset_name):
        with h5py.File(self.filename, "r") as file:
            return file[dataset_name][:]

    def get_keys(self, *args):
        largs = len(args)
        assert largs <= 1, f"Expected 0 or 1 arguments, received {largs}"
        if len(args) == 0:
            with h5py.File(self.filename, "r") as file:
                return list(file.keys())
        else:
            with h5py.File(self.filename, "r") as file:
                return list(file[args[0]].keys())
            
    def get_group_keys(self, group):
        """depricated now"""
        # deprication warning
        logging.warning("h5_logger.get_group_keys() is depricated. Use h5_logger.get_keys() instead.")
        with h5py.File(self.filename, "r") as file:
            return list(file[group].keys())
        
    def get_multiple(self, given_keys):
        with h5py.File(self.filename, "r") as file:
            return {k: file[k][()] for k in given_keys}

    def get_group(self, group_name):
        with h5py.File(self.filename, "r") as file:
            results = {}
            for key in file[group_name].keys():
                if isinstance(file[group_name][key], h5py.Dataset):
                    results[key] = file[group_name][key][()]
                elif isinstance(file[group_name][key], h5py.Group):
                    results[key] = self.get_group(f"{group_name}/{key}")
            return results
    
    def check_key(self, key):
        with h5py.File(self.filename, "r") as file:
            try:
                file[key]
                return True
            except KeyError:
                return False
    
    def get_group_keys(self, group):
        with h5py.File(self.filename, "r") as file:
            return list(file[group].keys())
        
    def rm_key(self, key):
        with h5py.File(self.filename, "r+") as file:
            del file[key]
            
    def move_key(self, key, new_key):
        with h5py.File(self.filename, "r+") as file:
            file.move(key, new_key)
            
    def move_group(self, source, destination):
        with h5py.File(self.filename, "r+") as file:
            file.move(source, destination)
            
    def get_unique_key(self, base_key):
        counter = 0
        key = base_key
        if base_key.endswith("/"):
            base_key = base_key[:-1]
            suffix = "/"
        else:
            suffix = ""
            
        while self.check_key(key):
            counter += 1
            key = f"{base_key}_{counter}{suffix}"
        return key
    
    def append_group_name(self, group_header, suffix=None):
        if suffix is None:
            suffix = "_"
        else:
            suffix = f"_{suffix}"
        if group_header.endswith("/"):
            header_parts = group_header.split("/")[:-1]
        else:
            header_parts = group_header.split("/")
        new_base_name = "/".join(header_parts) + suffix
        unique_base_name = self.get_unique_key(new_base_name)
        self.move_group(group_header, unique_base_name)
    
def check_if_in_h5(path, key):
    # check that file exists
    if not os.path.exists(path):
        return False
    with h5py.File(path, "r") as f:
        try:
            f[key]
            return True
        except KeyError:
            return False

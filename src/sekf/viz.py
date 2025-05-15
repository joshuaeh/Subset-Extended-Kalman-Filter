import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .utils import get_ci

def draw_axes(ax, linewidth=0.5, color="black"):
    """Draws the axes on the given matplotlib axis."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.hlines(0, xlim[0], xlim[1], color=color, linewidth=linewidth)
    ax.vlines(0, ylim[0], ylim[1], color=color, linewidth=linewidth)
    # reset the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

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

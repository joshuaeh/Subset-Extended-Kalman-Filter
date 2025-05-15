import numpy as np
import scipy.stats as stats

bytes_dict = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}

def get_ci(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n-1)  # get the t-score corresponding to the confidence interval
    return m, h

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

def numpy_memory_size(numpy_array, units="MB"):
    """Get the memory size of a numpy array"""
    return numpy_array.nbytes / bytes_dict[units]
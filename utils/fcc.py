import os
import time

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as skp 
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

from .__init__ import check_if_in_h5, default_rng, device
from .modeling import Exogenous_RkRNN, get_jacobian, get_parameter_gradient_vector, mask_fn
from .optimizers import GEKF, maskedAdam

NOC_STABLE_PATH = os.path.join("archive","Fluid-Catalytic-Cracking-Unit-Dataset-for-Process-Monitoring-Evaluation","NOC_stableFeedFlow_outputs.csv")
NOC_VARYING_PATH = os.path.join("archive","Fluid-Catalytic-Cracking-Unit-Dataset-for-Process-Monitoring-Evaluation","NOC_varyingFeedFlow_outputs.csv")
FAULTY_HX_PATH = os.path.join("archive","Fluid-Catalytic-Cracking-Unit-Dataset-for-Process-Monitoring-Evaluation","UAf_decrease_outputs.csv")
FAULTY_CONDENSER_PATH = os.path.join("archive","Fluid-Catalytic-Cracking-Unit-Dataset-for-Process-Monitoring-Evaluation","condEff_decrease_outputs.csv")
FAULTY_NAPTHASENSOR_PATH = os.path.join("archive","Fluid-Catalytic-Cracking-Unit-Dataset-for-Process-Monitoring-Evaluation","Fhn_sensorDrift_outputs.csv")
FAULTY_PRESSUREDROP_PATH = os.path.join("archive","Fluid-Catalytic-Cracking-Unit-Dataset-for-Process-Monitoring-Evaluation","deltaP_increase_outputs.csv")
FAULTY_CABVALVE_PATH = os.path.join("archive","Fluid-Catalytic-Cracking-Unit-Dataset-for-Process-Monitoring-Evaluation","CAB_valveLeak_outputs.csv")

COLUMNS = ["minutes",
    "F.fresh_feed", "T.ambient", "T.feed", "P.reactor",
    "P.delta", "P.regenerator", "F.air", "T.firebox",
    "T.preheated_feed", "T.reactor_riser", "T.regenerator", "L.standpipe",
    "T.stack_gas", "T.delta", "C.CO", "C.O2",
    "P.fractionator_overhead", "V.fractionator_pressure", "V.regenerator_temperature", "V.regenerator_pressure",
    "V.reactor_inventory", "V.preheated_feed_temperature", "V.reactor_temperature", "F.catalyst_regenerated",
    "F.catalyst_spent", "A.CAB", "A.WGC", "F.furnace_fuel", 
    "F.combustion_air", "F.stack_gas", "F.WGC_suction_valve", "P.CAB_suction",
    "P.CAB_discharge", "F.LPG", "F.LN", "F.HN",
    "F.LCO", "F.slurry", "F.reflux", "T.fractionator_overhead",
    "T.fractionator_mid", "T.fractionator_bottom", "V.accumulator_level", "V.fractionator_temperature",
    "V.HN_temperature", "V.LN_temperature"
]

SHORT_NAMES_MAP = {
    "regenerator" : "regen.",
    "reactor" : "react.",
    "preheated" : "preh.",
    "fractionator" : "fract.",
    "overhead" : "overh.",
    "bottom" : "bot.",
    "pressure": "press.",
    "temperature": "temp.",
    "inventory": "inv.",
    "catalyst" : "cat.",
    "combustion" : "comb.",
    "suction" : "suct.",
    "accumulator" : "accum."
}

FAULTY_CONDITIONS = {
        "Heat Exchanger Fouling" : {
            "path" : FAULTY_HX_PATH, 
            "variables" : [
                "F.furnace_fuel", 
                "T.firebox", 
                "V.preheated_feed_temperature"]},
        "Decreased Condenser Efficiency" : {"path" : FAULTY_CONDENSER_PATH, "variables" : ["F.LN", "F.LPG", "F.reflux", "F.WGC_suction_valve", "T.fractionator_overhead", "A.WGC", "V.accumulator_level", "V.fractionator_pressure"]},
        "Heavy Naptha Flow Sensor Drift" : {"path" : FAULTY_NAPTHASENSOR_PATH, "variables" : ["F.HN"]},
        "Higher Pressure Drop" : {"path" : FAULTY_PRESSUREDROP_PATH, "variables" : ["L.standpipe", "P.delta", "P.reactor", "V.reactor_inventory", "V.reactor_temperature"]},
        "CAB Valve Leak" : {"path" : FAULTY_CABVALVE_PATH, "variables" : ["V.regenerator_temperature"]}
    }

def get_short_name(name):
    for key, value in SHORT_NAMES_MAP.items():
        name = name.replace(key, value)
    return name

PRETTY_NAMES_MAP = {
    "A." : "Amperage: ",
    "C." : "Concentration: ",
    "F." : "Flow: ",
    "L." : "Level: ",
    "P." : "Pressure: ",
    "T." : "Temperature: ",
    "V." : "Valve: ",
    "_": " ",
}

def get_pretty_name(name):
    for key, value in PRETTY_NAMES_MAP.items():
        name = name.replace(key, value)
    return " ".join([w.title() if w.islower() else w for w in name.split()])

colors = sns.color_palette("hls", 7).as_hex()
color_map = {
        "A." : colors[1],
        "C." : colors[2],
        "F." : colors[3],
        "L." : colors[6],
        "P." : colors[4],
        "T." : colors[0],
        "V." : colors[5]
    }

U_COLUMNS = [1,2,3,18,19,20,21,22,23,26,27,43,44,45,46]
X_COLUMNS = [i for i in range(1, 47) if i not in U_COLUMNS]
U_COLUMNS_NAMES = [COLUMNS[i] for i in U_COLUMNS]
X_COLUMNS_NAMES = [COLUMNS[i] for i in X_COLUMNS]
sorted_columns = sorted(X_COLUMNS_NAMES) + sorted(U_COLUMNS_NAMES)

OPERATING_LIMITS = {
    "T.preheated_feed": (609, 629),  # a: (609, 629) c: (615, 625) 
    "T.reactor": (962, 975),
    "L.reactor": (94_500, 98_500),
    "T.regenerator": (1_242, 1_256),
    "P.regenerator": (27.5, 29.2),  # a: (27.5, 29.2) c: (27.5, 29.7)
    "P.fractionator_overhead": (24.6, 26),
    "L.accumulator": (55, 75),  # a: (55, 75) c: (60, 75)
    "T.fractionator_overhead": (242.33, 253.13),
    "T.HN": (520, 542),  # a: (520, 542) c: (523.13, 541.13)
    "T.LCO": (740, 761),  # a: (740, 761) c: (742.73, 760.73)
    "F.condensor_coolant": (10.98, 10.98),
    "D.PA_mid": (487.34, 537.10),
    "D.PA_bottom": (3_413.10, 3_443.75),
    "V.fractionator_pressure": (5, 95), 
    "V.regenerator_temperature": (5, 95), 
    "V.regenerator_pressure": (5, 95),
    "V.reactor_inventory": (5, 95), 
    "V.preheated_feed_temperature": (5, 95), 
    "V.reactor_temperature": (5, 95),
    "V.accumulator_level": (5, 95), 
    "V.fractionator_temperature": (5, 95),
    "V.HN_temperature": (5, 95), 
    "V.LN_temperature": (5, 95),
    "A.CAB": (50, 305),
    "A.WGC": (50, 230),
}

class StandardScaler():
    def __init__(self, center=True, scale=True):
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        
    def fit(self, X, variance_threshold=1e-10):
        self.center_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        self.constant_indices = np.where(self.var_ <= variance_threshold)[0]
        self.scale_ = self.var_.copy()
        self.scale_[self.constant_indices] = variance_threshold
        
    def transform(self, X):
        return (X - self.center_) / np.sqrt(self.scale_)
    
    def inverse_transform(self, X):
        return X * np.sqrt(self.scale_) + self.center_
    
    def fit_transform(self, X, variance_threshold=1e-5):
        self.fit(X, variance_threshold)
        return self.transform(X)

class FCUDataset(Dataset):
    def __init__(self,
            df: pd.DataFrame,
            prediction_horizon: int=60,
            context_length: int=1,
            X_columns:list=X_COLUMNS,
            U_columns:list=U_COLUMNS,
            train=False,
            X_scaler=None,
            U_scaler=None,
            begin_split_index=0,
            end_split_index=-1
            ):
        self.prediction_horizon = prediction_horizon
        self.context_length = context_length
        self.X_columns = X_columns
        self.U_columns = U_columns
        self.train = train
        self.X_scaler = X_scaler
        self.U_scaler = U_scaler
        self.begin_split_index = begin_split_index
        self.end_split_index = end_split_index
        
        self.transform(df)
        return
        
    def __len__(self):
        return len(self.X) - self.prediction_horizon - self.context_length

    def __getitem__(self, idx):
        return {
            "X0": torch.tensor(self.X[idx:idx+self.context_length], dtype=torch.float32),
            "U": torch.tensor(self.U[idx:idx+self.prediction_horizon], dtype=torch.float32),
            "X1": torch.tensor(self.X[idx+self.context_length:idx+self.context_length+self.prediction_horizon], dtype=torch.float32)
        }
    
    def transform(self, df):
        if self.X_scaler is not None:
            if self.train:
                self.X = self.X_scaler.fit_transform(df.iloc[self.begin_split_index:self.end_split_index, self.X_columns].values)
            else:
                self.X = self.X_scaler.transform(df.iloc[self.begin_split_index:self.end_split_index, self.X_columns].values)
        if self.U_scaler is not None:
            if self.train:
                self.U = self.U_scaler.fit_transform(df.iloc[self.begin_split_index:self.end_split_index, self.U_columns].values)
            else:
                self.U = self.U_scaler.transform(df.iloc[self.begin_split_index:self.end_split_index, self.U_columns].values)
        return
    
    def inverse_transform(self, X=None, U=None):
        assert X is not None or U is not None, "Either X or Y must be provided"
        if X is not None:
            x_shape = X.shape
            X = X.reshape(-1, X.shape[-1])
            X_ = self.X_scaler.inverse_transform(X)
            X_ = X_.reshape(x_shape)
        if U is not None:
            u_shape = U.shape
            U = U.reshape(-1, U.shape[-1])
            U_ = self.U_scaler.inverse_transform(U)
            U_ = U_.reshape(u_shape)
        if X is not None and U is not None:
            return X_, U_
        else:
            return X_ if X is not None else U_


def flatten_predictions(model, dataloader):
    """Rescale model inputs, outputs, and targets.
    ARGS:
        model: torch.nn.Module
        dataloaders: torch.utils.data.DataLoader NOTE: dataset must be FCUDataset so that it has the dataloader.dataset.inverse_transform method
        with U and X as kwargs
    RETURNS:
        X0: numpy.ndarray
        U: numpy.ndarray
        Y: numpy.ndarray
        YP: numpy.ndarray"""
    X0 = []
    U = []
    Y = []
    YP = []

    # iterate over dataloader
    for i, data in enumerate(dataloader):
        x0, u, x1 = data["X0"],data["U"], data["X1"]
        yp = model(x0, u)
        
        # concatenate
        X0 = np.concatenate([X0, x0.cpu().detach().numpy()], axis=0) if len(X0) > 0 else x0.cpu().detach().numpy()
        U = np.concatenate([U, u.cpu().detach().numpy()], axis=0) if len(U) > 0 else u.cpu().detach().numpy()
        Y = np.concatenate([Y, x1.cpu().detach().numpy()], axis=0) if len(Y) > 0 else x1.cpu().detach().numpy()
        YP = np.concatenate([YP, yp.cpu().detach().numpy()], axis=0) if len(YP) > 0 else yp.cpu().detach().numpy()
        
    # rescale
    X0 = dataloader.dataset.inverse_transform(X=X0)
    U = dataloader.dataset.inverse_transform(U=U)
    Y = dataloader.dataset.inverse_transform(X=Y)
    YP = dataloader.dataset.inverse_transform(X=YP)
    
    return X0, U, Y, YP

def graph_subplots(df, figsize=None):
    with sns.plotting_context("notebook", font_scale=0.8):
        if figsize is None:
            figsize = (13.3, 6.66)
            
        fig, axs = plt.subplots(12, 4, figsize=figsize)
        
        x_minor_ticks = np.arange(0, df.shape[0], 1440/2)
        x_major_ticks = np.arange(1440, df.shape[0], 1440)
        x_major_labels = [int(i) for i in np.arange(len(x_major_ticks))+1]

        # iterate over axs and set ylabel to sorted_columns value
        for i, ax in enumerate(axs.flatten()):
            if i >= len(sorted_columns)+1 or i == 31:
                ax.axis('off')
                continue
            if i > 31:
                i -= 1
            if i in [27, 42, 43, 44, 45, 46]:
                ax.set_xticks(x_major_ticks, labels=x_major_labels)
                ax.set_xticks(x_minor_ticks, minor=True)
            else:
                ax.set_xticks(x_major_ticks)
                ax.set_xticks(x_minor_ticks, minor=True)
                ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
            ax.plot(df[sorted_columns[i]], color=color_map[sorted_columns[i][:2]])
            ax.vlines(1440*2, df[sorted_columns[i]].min(), df[sorted_columns[i]].max(), color='k', linestyles='dashed', linewidth=0.5)
            ax.set_title(get_pretty_name(sorted_columns[i]), fontsize=12, fontweight="bold", y=1.0, pad=2)
            ax.set_xlim(0, df.shape[0])
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))

        # legend_entries = [mpl.lines.Line2D([0], [0], color=color_map[sorted_columns[i][:2]], label=get_pretty_name(sorted_columns[i])) for i in range(len(sorted_columns))]
        # legend_lines = []
        # legend_labels = []
        # for k,v in color_map.items():
        #     legend_lines.append(mpl.lines.Line2D([0], [0], color=v))
        #     legend_labels.append(PRETTY_NAMES_MAP[k][:-2])

        # axs[-1, 1].legend(legend_lines, legend_labels, ncol=4, bbox_to_anchor=(0.6, 0), loc='upper left', fontsize=12)
        
        # fig.text(0.5, 1.02, ha="center", va="center", s="Normal Operating Conditions FCC Data", fontsize=24, fontweight="bold")
        fig.text(-0.00, 8/12, ha="center", va="center", rotation=90, s="Process Variables", fontsize=16, fontweight="bold")
        fig.text(-0.00, 2/12, ha="center", va="center", rotation=90, s="Exogenous Inputs", fontsize=16, fontweight="bold")
        # fig.text(0.875, 4.5/12, ha="center", va="center", s="Disturbances", fontsize=16, fontweight="bold")
        fig.text(0.5, 0, ha="center", s="Training Data Day", fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.patches.extend([plt.Rectangle(xy=(-0.02,-0.02), width=1.02, height=4.5/12, fill=True, color="lightgrey", zorder=-1, transform=fig.transFigure, figure=fig)])
        # fig.patches.extend([plt.Rectangle(xy=(2.02/4,3.35/12), width=1.98/4, height=0.92/12, fill=True, color="sandybrown", zorder=-1, transform=fig.transFigure, figure=fig)])
        # fig.patches.extend([plt.Rectangle(xy=(3.02/4,4.0/12), width=0.98/4, height=0.8/12, fill=True, color="sandybrown", zorder=-1, transform=fig.transFigure, figure=fig)])
        fig.subplots_adjust(hspace=0.7)
    
    return fig, axs

def plot_prediction(data, j=None, X_scaler=None, U_scaler=None, figsize=(16, 9)):
    X0, U, Y, YP = data

    if j is None:
        j = default_rng.integers(0, Y.shape[0], 1)[0]
    
    x0 = X0[j, :, :]
    y = Y[j, :, :]
    yp = YP[j, :, :]
    u = U[j, :, :]
    
    if X_scaler is not None:
        y = X_scaler.transform(y)
        yp = X_scaler.transform(yp)
        x0 = X_scaler.transform(x0)
        nMAE = np.abs(y-yp).mean()
        x_prefix = "Scaled "
        units = "Scaled "
    else:
        nMAE = np.abs(y-yp).mean() / np.abs(y).mean()
        x_prefix = ""
        units = "n"
    if U_scaler is not None:
        u = U_scaler.transform(u)
        u_prefix = "Scaled "
    else:
        u_prefix = ""

    fig, axs = plt.subplots(12, 4, figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        x_coords = np.arange(y.shape[0])+1
        # X
        if i < 31:
            # X
            label = sorted(X_COLUMNS_NAMES)[i]
            label_index = X_COLUMNS_NAMES.index(label)
            ax.plot(x_coords, y[:, label_index], "b-")
            ax.plot(x_coords, yp[:, label_index], "r:")
            ax.scatter(0, x0[0, label_index], c="b", marker="o")
            ax.set_title(label)
            
        elif (i > 31) and (i < 46):
            # U
            ii = i - 32
            label = sorted(U_COLUMNS_NAMES)[ii]
            label_index = U_COLUMNS_NAMES.index(label)
            ax.plot(u[:, label_index], "k-")
            ax.set_title(label)
        
        else:
            ax.axis('off')
            continue
        
    fig.text(-0.00, 8/12, ha="center", va="center", rotation=90, s=f"{x_prefix}Process Variables", fontsize=16, fontweight="bold")
    fig.text(-0.00, 2/12, ha="center", va="center", rotation=90, s=f"{u_prefix}Exogenous Inputs", fontsize=16, fontweight="bold")
    fig.text(0.5, 0, ha="center", va="center", s="Prediction Horizon", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.7/12, ha="left", va="center", s=f"Prediction index: {j:,} of {Y.shape[0]:,}", fontsize=14)
    fig.text(0.5, 0.5/12, ha="left", va="center", s=f"Prediction {units}MAE: {nMAE:4g}", fontsize=14)

    fig.tight_layout()
    fig.patches.extend([plt.Rectangle(xy=(-0.01,-0.2/12), width=1.01, height=4.2/12, fill=True, color="lightgrey", zorder=-1, transform=fig.transFigure, figure=fig)])
        
    # fig.subplots_adjust(hspace=0.9)
    fig.legend(["True", "Predicted"], bbox_to_anchor=(3.1/4,5/12), loc="upper left", ncol=1)
    fig.tight_layout()
    
    return fig, axs

def plot_nmae_horizon(data, X_scaler=None, figsize=(16, 9)):
    X0, U, Y, YP = data
    if X_scaler is not None:
        Y = X_scaler.transform(Y.reshape(-1, Y.shape[-1])).reshape(Y.shape)
        YP = X_scaler.transform(YP.reshape(-1, YP.shape[-1])).reshape(YP.shape)
        nmae = np.abs(Y - YP).mean(axis=0)
    else:
        mae = np.abs(Y - YP).mean(axis=0)
        nmae = mae / np.abs(Y).mean(axis=0)
    fig, axs = plt.subplots(8, 4, figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        if i >= 31:
            ax.axis('off')
            continue
        
        # X
        label = sorted(X_COLUMNS_NAMES)[i]
        label_index = X_COLUMNS_NAMES.index(label)
        ax.plot(nmae[:, label_index], "b-")
        ax.set_title(label)
        ax.set_ylim(0, nmae.max()*1.05)
    if X_scaler is not None:
        x_prefix = "Scaled "
    else:
        x_prefix = "n"
    fig.text(0.76, 0.7/12, ha="left", va="center", s=x_prefix+f"MAE: {nmae.mean():.6f}", fontsize=14)
    # fig.subplots_adjust(hspace=0.9)
    fig.tight_layout()
    return fig, axs

rk_trained_weights_path = os.path.join("results","fcc", "training", "RkRNN_NOC.pth")

def sefk_updating_trial(faulty_condition_name, 
    # parameter selection arguments                    
    thresh=None, quantile_thresh=None, 
    # updating arugments
    lr=1e-5, q=0.1, p0=100, X_scaler=None, U_scaler=None, measurement_noise_level=0.0025,
    # logging arguments
    find_unique_header=False, logger=None):
    # initialize, validate aruments
    assert logger is not None, "logger must be provided"
    assert X_scaler is not None, "X_scaler must be provided"
    assert U_scaler is not None, "U_scaler must be provided"
    
    if log_header is None:
        if thresh is not None:
            log_header = f"updating/{faulty_condition_name}/t-{thresh:.2f}/"
        elif quantile_thresh is not None:
            log_header = f"updating/{faulty_condition_name}/q-{quantile_thresh:.2f}/"
        else:
            log_header = f"updating/{faulty_condition_name}/full/"
    if find_unique_header:
        log_header = logger.get_unique_key(log_header)
    if not logger.check_key(log_header+"YP"):
        data = logger.get_dataset(f"updating/{faulty_condition_name}/data/df_data")[0]
        df = pd.DataFrame(data, columns=COLUMNS)
        faulty_dataset = FCUDataset(df,
            X_scaler=X_scaler,
            U_scaler=U_scaler)
        faulty_dataloader = DataLoader(faulty_dataset, batch_size=1, shuffle=False, generator=torch.Generator(device=device))

        model = Exogenous_RkRNN(state_dim=len(X_COLUMNS), input_dim=len(U_COLUMNS), hidden_size=64)
        model.load_state_dict(torch.load(rk_trained_weights_path, map_location=device, weights_only=True))

        optimizer = GEKF(model.parameters(), lr=lr, q=q, p0=p0)
        criterion = torch.nn.MSELoss()

        logger.log_dict({
            log_header+"parameters/lr": np.array([lr]),
            log_header+"parameters/q": np.array([q]),
            log_header+"parameters/p0": np.array([p0]),
            log_header+"parameters/noise_level": np.array([measurement_noise_level]),
        })

        start_time = time.time()

        model.train()
        pbar = tqdm(faulty_dataloader, desc=log_header)
        for data in pbar:
            optimizer.zero_grad()
            X0, U, X1 = data["X0"], data["U"], data["X1"]
            with torch.no_grad():
                j = get_jacobian(model, (X0, U))
            X_pred = model(X0, U)
            innovation = X1 - X_pred
            loss = criterion(X_pred, X1)
            loss.backward()
            grads = get_parameter_gradient_vector(model)
            mask = mask_fn(grads, thresh=thresh, quantile_thresh=quantile_thresh)
            optimizer.step(innovation, j, mask)
            
            logger.log_dict({
                log_header+"X0": X0.detach().cpu().numpy(),
                log_header+"U": U.detach().cpu().numpy(),
                log_header+"Y": X1.detach().cpu().numpy(),
                log_header+"YP": X_pred.detach().cpu().numpy(),
                log_header+"innovation": innovation.detach().cpu().numpy(),
                log_header+"loss": np.mean(loss.item()),
                log_header+"mask": mask.detach().cpu().numpy(),
                log_header+"weights": optimizer._get_flat_params().detach().cpu().numpy(),
                log_header+"time": np.array([time.time() - start_time])
            })
            
            pbar.set_description(f"{log_header} Loss: {np.mean(loss.item()):.4f} avg loss: {np.mean(logger.get_dataset(log_header+'loss')):.4f}")
            
def retraining_trial(faulty_condition_name,
    # parameter selection arguments
    thresh=None, quantile_thresh=None, 
    # retraining arguments
    loss_threshold=0.2, moving_horizon_length=50, retraining_batch_size=5, retraining_epochs=50, lr=1e-5, 
    early_stopping_threshold=0, divergence_threshold=99, X_scaler=None, U_scaler=None,
    measurement_noise_level=0.0025,
    # logging arguments
    log_header=None, logger=None, find_unique_header=False):
    
    ### initialize ###
    assert logger is not None, "logger must be provided"
    assert X_scaler is not None, "X_scaler must be provided"
    assert U_scaler is not None, "U_scaler must be provided"
    
    if log_header is None:
        if thresh is not None:
            log_header = f"retraining/{faulty_condition_name}/t-{thresh:.2f}/"
        elif quantile_thresh is not None:
            log_header = f"retraining/{faulty_condition_name}/q-{quantile_thresh:.2f}/"
        else:
            log_header = f"retraining/{faulty_condition_name}/full/"
    if find_unique_header:
        log_header = logger.get_unique_key(log_header)
    if not logger.check_key(log_header+"YP"):
        data = logger.get_dataset(f"updating/{faulty_condition_name}/data/df_data")[0]
        df = pd.DataFrame(data, columns=COLUMNS)
        faulty_dataset = FCUDataset(df,
            X_scaler=X_scaler,
            U_scaler=U_scaler)
        faulty_dataloader = DataLoader(faulty_dataset, batch_size=1, shuffle=False, generator=torch.Generator(device=device))

        model = Exogenous_RkRNN(state_dim=len(X_COLUMNS), input_dim=len(U_COLUMNS), hidden_size=64)
        model.load_state_dict(torch.load(rk_trained_weights_path, map_location=device, weights_only=True))

        optimizer = maskedAdam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        logger.log_dict({
            log_header+"parameters/lr": np.array([lr]),
            log_header+"parameters/noise_level": np.array([measurement_noise_level]),
            log_header+"parameters/loss_threshold": np.array([loss_threshold]),
            log_header+"parameters/moving_horizon_length": np.array([moving_horizon_length]),
            log_header+"parameters/retraining_epochs": np.array([retraining_epochs]),
            log_header+"parameters/early_stopping_threshold": np.array([early_stopping_threshold]),
            log_header+"parameters/divergence_threshold": np.array([divergence_threshold])
        })

        start_time = time.time()

        model.train()
        pbar = tqdm(enumerate(faulty_dataloader), total=faulty_dataset.__len__(), desc=log_header)
        ### Training iterating through dataset ###
        for i, data in pbar:
            optimizer.zero_grad()
            X0, U, X1 = data["X0"], data["U"], data["X1"]
            X_pred = model(X0, U)
            loss = criterion(X_pred, X1)
            loss.backward()
            grads = get_parameter_gradient_vector(model)
            mask = mask_fn(grads, thresh=thresh, quantile_thresh=quantile_thresh)
            
            logger.log_dict({
                log_header+"X0": X0.detach().cpu().numpy(),
                log_header+"U": U.detach().cpu().numpy(),
                log_header+"Y": X1.detach().cpu().numpy(),
                log_header+"YP": X_pred.detach().cpu().numpy(),
                # log_header+"innovation": innovation.detach().cpu().numpy(),
                log_header+"loss": np.mean(loss.item()),
                log_header+"grads": grads.detach().cpu().numpy(),
                log_header+"mask": mask.detach().cpu().numpy(),
                log_header+"weights": optimizer._get_flat_params().detach().cpu().numpy(),
                # log_header+"time": np.array([time.time() - start_time])
            })
            
            ### retraining ###
            if loss.item() > loss_threshold:
                # reinitialize optimizer to reset momentum
                optimizer = maskedAdam(model.parameters(), lr=lr)
                
                moving_horizon_start = i - moving_horizon_length if i > moving_horizon_length else 0
                retraining_indices = np.arange(moving_horizon_start, i)
                # retraining_losses = []
                retraining_sampler = torch.utils.data.SubsetRandomSampler(retraining_indices, generator=torch.Generator(device=device))
                retraining_dataloader = DataLoader(faulty_dataset, batch_size=retraining_batch_size, sampler=retraining_sampler)
                for retraining_epoch in range(retraining_epochs):
                    for retraining_batch in retraining_dataloader:
                        optimizer.zero_grad()
                        X0, U, X1 = retraining_batch["X0"], retraining_batch["U"], retraining_batch["X1"]
                        X_pred = model(X0, U)
                        retraining_loss = criterion(X_pred, X1)
                        retraining_loss.backward()
                        optimizer.masked_step(mask=mask)
                        # retraining_losses.append(retraining_loss.item())
                        if retraining_loss.item() < early_stopping_threshold:
                            break
                    if retraining_loss.item() < early_stopping_threshold:
                        break
                # logger.log_dict({
                #     log_header+"retraining_loss": np.array(retraining_losses)
                # })
            
            # log time after retraining
            logger.log_dict({
                log_header+"time": np.array([time.time() - start_time])
            })
            
            pbar.set_description(f"{log_header} Loss: {np.mean(loss.item()):.4f}")
            
            ### check for divergence ###
            if (loss.item() > divergence_threshold) or np.isnan(loss.item()):
                print(f" {log_header} Diverged ".center(40, "="))
                logger.append_group_name(log_header, suffix="diverged")
                return

def get_updating_stats(logger):
    method_headers = ["full", "t-0.99", "t-0.95", "q-0.99", "q-0.95"]
    faulty_conditions = FAULTY_CONDITIONS.keys()
    rows = []
    for fc in faulty_conditions:
        row = {}
        row["faulty condition"] = fc
        for method in method_headers:
            logger_header = f"updating/{fc}/{method}/"
            if not logger.check_key(logger_header+"loss"):
                row[method] = "NA"
            else:
                row[method] = logger.get_dataset(logger_header+"loss").mean()
        rows.append(row)
    return pd.DataFrame(rows)

def get_retraining_stats(logger):
    # TODO account for multiple converged iterations
    method_headers = ["full", "t-0.99", "t-0.95", "q-0.99", "q-0.95"]
    faulty_conditions = FAULTY_CONDITIONS.keys()
    rows = []
    for fc in faulty_conditions:
        row = {}
        row["faulty condition"] = fc
        faulty_condition_experiments = logger.get_keys(f"retraining/{fc}/")
        for method in method_headers:
            logger_header = f"retraining/{fc}/{method}/"
            # check if it exists and first dimension of results = 1378
            if not logger.check_key(logger_header+"loss"):
                row[method] = "NA"
            elif logger.get_dataset(logger_header+"loss").shape[0] != 1378:
                row[method] = f"Incomplete ({logger.get_dataset(logger_header+'loss').shape[0]}/1378)"
            else:
                row[method] = logger.get_dataset(logger_header+"loss").mean()
        rows.append(row)
    return pd.DataFrame(rows)

def get_stats(logger,
    maintenance_methods = ["updating", "retraining"]):
    parameter_selection_methods = ["full", "t-0.99", "t-0.95", "q-0.99", "q-0.95"]
    faulty_conditions = list(FAULTY_CONDITIONS.keys())
    data = []
    for mm in maintenance_methods:
        for fc in faulty_conditions:
            fc_experiments = logger.get_keys(f"{mm}/{fc}/")
            for selection_method in parameter_selection_methods:
                row = {
                    "maintenance_method": mm,
                    "faulty_condition": fc,
                    "selection method": selection_method,
                    "loss": 999,
                    "time": 999,
                    "attempts": 0,
                    "best result": "NA"
                }
                matching_experiments = [e for e in fc_experiments if e.startswith(selection_method)]
                row["attempts"] = len(matching_experiments)
                for me in matching_experiments:
                    logger_header = f"{mm}/{fc}/{me}/"
                    loss_data = logger.get_dataset(logger_header+"loss")
                    time_data = logger.get_dataset(logger_header+"time")
                    if loss_data.shape[0] == 1378:
                        if np.isnan(loss_data).sum() == 0:
                            if loss_data.mean() < row["loss"]:
                                row["loss"] = loss_data.mean()
                                row["time"] = time_data.mean()
                                row["best result"] = me
                data.append(row)
    return pd.DataFrame(data)
                
                
                
            
    
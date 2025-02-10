"""Blood Glucose utility functions."""
import datetime
import os
import time
from typing import Optional, Union

from labellines import labelLine
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import tqdm
import torch

from .__init__ import default_rng, device, h5_logger
from .modeling import Exogenous_RkRNN, EarlyStopper
from .modeling import get_jacobian, get_parameter_gradient_vector, init_weights, mask_fn
from .optimizers import GEKF, maskedAdam

device = torch.device("cpu")

DATA_STEP_SIZE = 10

class DiabeticPatient():
    def __init__(self, 
        random_params=False,
        param_kwargs=None,
        meal_kwargs=None,
        ):
        self.rng = default_rng
        self.define_parameters(random_params=random_params, param_kwargs=param_kwargs)
        self.define_meal_parameters(meal_kwargs=meal_kwargs)
        pass
    
    def define_parameters(self, random_params=False, param_kwargs=None): 
        params = {
            'p1': 1.57e-2,
            'p1_std': 1e-3,
            'Gb': 100,
            'Gb_std': 7.9,
            'p2': 1.23e-2,
            'p2_std': 4.3e-4,
            'si': 5.0e-1,
            'si_std': 1.1e-2,
            'ke': 1.82e-2,
            'ke_std': 5.2e-4,
            'kabs': 1.2e-2,
            'kabs_std': 3.8e-4,
            'kemp': 1.8e-1,
            'kemp_std': 1.5e-2,
            'Vi': 12.0,
            'Vi_std': 3.0e-1,
            'fkabs_Vg': 8e-2,
            'fkabs_Vg_std': 2.0e-3,
            'Ib': 4e-2,
            'Ib_std': 2.0e-3,
        }
        if param_kwargs is not None:
            params.update(param_kwargs)
            
        if random_params:
            self.p1 = self.rng.normal(params['p1'], params['p1_std'])
            self.Gb = self.rng.normal(params['Gb'], params['Gb_std'])
            self.p2 = self.rng.normal(params['p2'], params['p2_std'])
            self.si = self.rng.normal(params['si'], params['si_std'])
            self.ke = self.rng.normal(params['ke'], params['ke_std'])
            self.kabs = self.rng.normal(params['kabs'], params['kabs_std'])
            self.kemp = self.rng.normal(params['kemp'], params['kemp_std'])
            self.Vi = self.rng.normal(params['Vi'], params['Vi_std'])
            self.fkabs_Vg = self.rng.normal(params['fkabs_Vg'], params['fkabs_Vg_std'])
            self.Ib = self.rng.normal(params['Ib'], params['Ib_std'])
        else:
            self.p1 = params['p1']
            self.Gb = params['Gb']
            self.p2 = params['p2']
            self.si = params['si']
            self.ke = params['ke']
            self.kabs = params['kabs']
            self.kemp = params['kemp']
            self.Vi = params['Vi']
            self.fkabs_Vg = params['fkabs_Vg']
            self.Ib = params['Ib']
        
        return
    
    def define_meal_parameters(self, meal_kwargs=None):
        # "On average, children and adults consume approximately 22 percent of their total daily calorie intake at
        # breakfast, 31 percent at lunch, 35 percent at dinner, and the remainder in snacks." (Institute of Medicine (US) Committee to Review Child and Adult Care Food Program Meal Requirements; Murphy SP, Yaktine AL, West Suitor C, et al., editors. Child and Adult Care Food Program: Aligning Dietary Guidance for All. Washington (DC): National Academies Press (US); 2011. 6, Process for Developing Recommendations for Meal Requirements. Available from: https://www.ncbi.nlm.nih.gov/books/NBK209815)
        meal_params = {
            'meal1_time': (6, 9),
            'meal1_size': (60, 20),
            'meal2_time': (11, 13),
            'meal2_size': (90, 30),
            'meal3_time': (17, 20),
            'meal3_size': (90, 30),
            
            "meal_effect_indices": np.array([0], dtype=int),
            "meal_effect_proportion": np.array([1]),
        }
        if meal_kwargs is not None:
            meal_params.update(meal_kwargs)
            
        self.meal1_time = meal_params['meal1_time']
        self.meal1_size = meal_params['meal1_size']
        self.meal2_time = meal_params['meal2_time']
        self.meal2_size = meal_params['meal2_size']
        self.meal3_time = meal_params['meal3_time']
        self.meal3_size = meal_params['meal3_size']
        self.meal_effect_indices = meal_params['meal_effect_indices']
        self.meal_effect_proportion = meal_params['meal_effect_proportion']
        
    def get_d(self, days):
        d = np.zeros(days * 1440 + 361)  # add 361: 1 for end point, 360 for tail of meal disturbance that will be cut off
        
        meal1_times = default_rng.integers(60*self.meal1_time[0], 60*self.meal1_time[1], size=days)
        meal1_sizes = default_rng.normal(self.meal1_size[0], self.meal1_size[1], size=days)
        meal1_start_indices = meal1_times + np.arange(days) * 1440
        meal1_indices = self.meal_effect_indices + meal1_start_indices[:, None]
        meal1_effect = self.meal_effect_proportion * meal1_sizes[:, None]
        meal2_times = default_rng.integers(60*self.meal2_time[0], 60*self.meal2_time[1], size=days)
        meal2_sizes = default_rng.normal(self.meal2_size[0], self.meal2_size[1], size=days)
        meal2_start_indices = meal2_times + np.arange(days) * 1440
        meal2_indices = self.meal_effect_indices + meal2_start_indices[:, None]
        meal2_effect = self.meal_effect_proportion * meal2_sizes[:, None]
        meal3_times = default_rng.integers(60*self.meal3_time[0], 60*self.meal3_time[1], size=days)
        meal3_sizes = default_rng.normal(self.meal3_size[0], self.meal3_size[1], size=days)
        meal3_start_indices = meal3_times + np.arange(days) * 1440
        meal3_indices = self.meal_effect_indices + meal3_start_indices[:, None]
        meal3_effect = self.meal_effect_proportion * meal3_sizes[:, None]

        d[meal1_indices] += meal1_effect
        d[meal2_indices] += meal2_effect
        d[meal3_indices] += meal3_effect

        return d[:-360] # cut off the tail
    
    def get_params(self):
        return self.p1, self.Gb, self.p2, self.si, self.ke, self.kabs, self.kemp, self.Vi, self.fkabs_Vg, self.Ib
    
    def dydt(self, t, y, u, d, dparams):
        """ 
        States (y):
        G: Blood Glucose (mg/dL)
        X: Remote Insulin (μU/mL)
        I: Plasma Insulin (μU/mL)
        Q1: Carb Mass in Gut 1 (mg)
        Q2: Carb Mass in Gut 2 (mg)
        G_gut: Glucose in Gut (mg)

        Exogenous Inputs:
        u: Insulin infusion (U/min)
        d: Meal disturbance (mg/min)
        """
        # States
        G, X, I, Q1, Q2, G_gut, Gb, si, Ib = y

        # parameter rate of change
        dGb, dsi, dIb = dparams

        # calculate the derivatives
        dGdt = -self.p1 * (G-Gb) - si*X*G + self.fkabs_Vg * G_gut
        dXdt = -self.p2 * (X-I-Ib)
        dIdt = u/self.Vi - self.ke*I
        dQ1dt = d - self.kemp*Q1
        dQ2dt = self.kemp*(Q1-Q2)
        dG_gutdt = self.kemp*Q2 - self.kabs*G_gut

        dGbdt = dGb * default_rng.uniform(0, 1)
        dsidt = dsi * default_rng.uniform(0, 1)
        dIbdt = dIb * default_rng.uniform(0, 1)

        # print(dGdt, dXdt, dIdt, dQ1dt, dQ2dt, dG_gutdt, dGbdt, dsidt, dIbdt)

        return np.array([dGdt, dXdt, dIdt, dQ1dt, dQ2dt, dG_gutdt, dGbdt, dsidt, dIbdt])
    
    def sim_patient(self, y0, t, u, d, dparams):
        Y = [y0]
        for i in tqdm(range(len(t)-1)):
            time_span = [t[i], t[i+1]]
            y = solve_ivp(self.dydt, time_span, Y[-1], args=(u[i], d[i], dparams), method='RK45', rtol=1e-6, atol=1e-9).y.T[-1]
            Y.append(y)

        Y = np.array(Y)
        return Y
    
    def sim_safe_patient(self, y0, t, u, d, dparams, insulin_limit=85, dangerous_low_glucose=70, dUdt_limit=0.016):
        Y = [y0]
        for i in tqdm.tqdm(range(len(t)-1)):
            time_span = [t[i], t[i+1]]
            if Y[-1][0] < insulin_limit:
                u[i] = max(0, u[i-1] - dUdt_limit)  # lower insulin rate by as much as possible
                u[i:i+45] = u[i]
            if Y[-1][0] < dangerous_low_glucose:
                d[i] = d[i] + 2
            if i > 0:
                if u[i] - u[i-1] > dUdt_limit:
                    u[i] = u[i-1] + dUdt_limit
            y = solve_ivp(self.dydt, time_span, Y[-1], args=(u[i], d[i], dparams), method='RK45', rtol=1e-6, atol=1e-9).y.T[-1]
            Y.append(y)

        Y = np.array(Y)
        return Y

class GlucoseDataset(torch.utils.data.Dataset):
    def __init__(self,
        data : dict[str, np.ndarray],
        prediction_horizon : Optional[int] = None,
        input_horizon : Optional[int] = None,
        step_size : Optional[int] = None,
        databounds : Optional[Union[list[float, float], tuple[float, float]]] = None,
        train: bool = False,
        Xscaler = None,
        Uscaler = None,
        ):
        """
        
        ARGS
        ---
        data : dict[str, np.ndarray]
            A dictionary containing the following keys: 'D', 'U', 'G', 'X', 'I', 'Q1', 'Q2', 'G_gut'
        prediction_horizon : Optional[int]
            The number of minutes into the future to predict
            DEFAULT : 60
        input_horizon : Optional[int]
            The number of minutes into the past to use as input
            DEFAULT : 1
        databounds : Optional[Union[list[float, float], tuple[float, float]]]
            The lower and upper bounds for the data
            DEFAULT : [0, 1]
        train : bool
            Whether or not to fit the scaler
            DEFAULT : False
        Xscaler : Optional[sklearn.preprocessing.Scaler]
            The scaler to use for the states
            DEFAULT : None
        Uscaler : Optional[sklearn.preprocessing.Scaler]
            The scaler to use for the targets
            DEFAULT : None
        """
        # parse through default arguments
        if prediction_horizon is None:
            prediction_horizon = 60
        self.prediction_horizon = prediction_horizon
        if input_horizon is None:
            input_horizon = 1
        self.input_horizon = input_horizon
        if step_size is None:
            step_size = 1
        self.step_size = step_size
        if databounds is None:
            databounds = [0, 1]
        self.databounds = databounds
        self.Xscaler = Xscaler
        self.Uscaler = Uscaler
        
        # data
        # TODO check that all values are the same size in data
        n_total_datapoints = len(data['G'])
        n_total_data_windows = (n_total_datapoints - self.prediction_horizon - self.input_horizon) // self.step_size
        idx_first_window = int(n_total_data_windows * self.databounds[0])
        idx_last_window = int(n_total_data_windows * self.databounds[1])
        
        self.n_windows = (idx_last_window - idx_first_window) // self.step_size
        self.n_X = 6
        self.n_U = 2
        
        # get data into one array
        self.Xdata = np.stack([
            data["G"][idx_first_window:idx_last_window + self.input_horizon + self.prediction_horizon],
            data["X"][idx_first_window:idx_last_window + self.input_horizon + self.prediction_horizon],
            data["I"][idx_first_window:idx_last_window + self.input_horizon + self.prediction_horizon],
            data["Q1"][idx_first_window:idx_last_window + self.input_horizon + self.prediction_horizon],
            data["Q2"][idx_first_window:idx_last_window + self.input_horizon + self.prediction_horizon],
            data["G_gut"][idx_first_window:idx_last_window + self.input_horizon + self.prediction_horizon],
        ], axis=-1)
        self.Udata = np.stack([
            data["U"][idx_first_window:idx_last_window + self.input_horizon + self.prediction_horizon],
            data["D"][idx_first_window:idx_last_window + self.input_horizon + self.prediction_horizon]
        ], axis=-1)
        
        # fit / scale data
        if train:
            self.fit(self.Xdata, self.Udata)
        if self.Xscaler is not None:
            self.Xdata = self.transform(self.Xdata, self.Xscaler)
        if self.Uscaler is not None:
            self.Udata = self.transform(self.Udata, self.Uscaler)
            
        # torch tensors
        self.Xdata = torch.tensor(self.Xdata, dtype=torch.float32, device=device)
        self.Udata = torch.tensor(self.Udata, dtype=torch.float32, device=device)
        return
            
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        _idx = idx * self.step_size
        return {
            "X0" : self.Xdata[_idx:_idx+self.input_horizon],
            "U" : self.Udata[_idx:_idx+self.prediction_horizon],
            "X" : self.Xdata[_idx+self.input_horizon:_idx+self.input_horizon+self.prediction_horizon]
        }
    
    def fit(self, Xdata, Udata):
        if self.Xscaler is not None:
            self.Xscaler.fit(Xdata)
        if self.Uscaler is not None:
            self.Uscaler.fit(Udata)
        return
    
    # TODO: Handle tensors and numpy arrays
    def transform(self, data, scaler):
        match data.ndim:
            case 1:
                return self._1d_transform(data, scaler)
            case 2:
                return self._2d_transform(data, scaler)
            case 3:
                return self._3d_transform(data, scaler)
    
    def _1d_transform(self, data, scaler):
        return scaler.transform(data.reshape(-1, 1)).reshape(-1)
    
    def _2d_transform(self, data, scaler):
        # sklearn expects the data 
        return scaler.transform(data)
    
    def _3d_transform(self, data, scaler):
        initial_shape = data.shape
        data = data.reshape(-1, data.shape[-1])
        data = scaler.transform(data)
        return data.reshape(initial_shape)
    
    # TODO torch / numpy
    def inverse_transform(self, data, scaler):
        match data.ndim:
            case 1:
                return self._1d_inverse_transform(data, scaler)
            case 2:
                return self._2d_inverse_transform(data, scaler)
            case 3:
                return self._3d_inverse_transform(data, scaler)
    
    def _1d_inverse_transform(self, data, scaler):
        return scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)
    
    def _2d_inverse_transform(self, data, scaler):
        return scaler.inverse_transform(data)
    
    def _3d_inverse_transform(self, data, scaler):
        initial_shape = data.shape
        data = data.reshape(-1, data.shape[-1])
        data = scaler.inverse_transform(data)
        return data.reshape(initial_shape)
    
    def eval_model(self, model, logger=None, logger_base_key=None, logger_replace=False):
        model.eval()
        with torch.no_grad():
            dl = torch.utils.data.DataLoader(self, batch_size=len(self), shuffle=False, generator=torch.Generator(device=device))
            data = next(iter(dl))
            X0, U, X = data["X0"], data["U"], data["X"]
            XP = model(X0, U)
            # convert to numpy arrays
            X0_scaled = X0.detach().cpu().numpy()
            U_scaled = U.detach().cpu().numpy()
            X_scaled = X.detach().cpu().numpy()
            XP_scaled = XP.detach().cpu().numpy()
            # inverse transform
            X0 = self.inverse_transform(X0_scaled, self.Xscaler)
            U = self.inverse_transform(U_scaled, self.Uscaler)
            X = self.inverse_transform(X_scaled, self.Xscaler)
            XP = self.inverse_transform(XP_scaled, self.Xscaler)
        if logger is not None:
            assert logger_base_key is not None, "logger_base_key must be provided if logger is provided"
            if not logger_base_key.endswith("/"):
                if len(logger_base_key)>0:
                    logger_base_key += "/"
            logger.log_attribute(logger_base_key +"x0_scaled", X0_scaled, replace=logger_replace)
            logger.log_attribute(logger_base_key +"u_scaled", U_scaled, replace=logger_replace)
            logger.log_attribute(logger_base_key +"x_scaled", X_scaled, replace=logger_replace)
            logger.log_attribute(logger_base_key +"xp_scaled", XP_scaled, replace=logger_replace)
            logger.log_attribute(logger_base_key +"x0", X0, replace=logger_replace)
            logger.log_attribute(logger_base_key +"u", U, replace=logger_replace)
            logger.log_attribute(logger_base_key +"x", X, replace=logger_replace)
            logger.log_attribute(logger_base_key +"xp", XP, replace=logger_replace)
        return X0, U, X, XP
    
def get_glucose_data(logger, base_key):
    return {
        'D': logger.get_dataset(base_key + "/D")[0],
        'U': logger.get_dataset(base_key + "/U")[0],
        'G': logger.get_dataset(base_key + "/G")[0],
        'X': logger.get_dataset(base_key + "/X")[0],
        'I': logger.get_dataset(base_key + "/I")[0],
        'Q1': logger.get_dataset(base_key + "/Q1")[0],
        'Q2': logger.get_dataset(base_key + "/Q2")[0],
        'G_gut': logger.get_dataset(base_key + "/G_gut")[0]
    }
    
def train_minibatch(model, X0, U, X, criterion, optimizer):
    optimizer.zero_grad()
    Y = model(X0, U)
    loss = criterion(Y, X)
    loss.backward()
    optimizer.step()
    return loss.item()

def val_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            X0 = batch['X0']
            U = batch['U']
            X = batch['X']

            Y = model(X0, U)
            loss = criterion(Y, X)

            running_loss += loss.item()
    return running_loss / len(dataloader)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, lr_scheduler, early_stopper, checkpointing=True):
    identifier_string = datetime.datetime.now().strftime("%Y.%m.%d %H.%M")
    train_losses = []
    val_losses = []
    lrs = []
    epoch = 0
    while not early_stopper():
        model.train()
        running_loss = 0
        for i, batch in enumerate(train_dataloader):
            X0 = batch['X0']
            U = batch['U']
            X = batch['X']
            
            loss = train_minibatch(model, X0, U, X, criterion, optimizer)
            running_loss += loss
            
            print(f"\rEpoch [{epoch+1}], Minibatch [{i+1}/{len(train_dataloader)}] Train: {0 if len(train_losses) == 0 else train_losses[-1]:.4g} Val: {0 if len(val_losses) == 0 else val_losses[-1]:.4g} lr: {lr_scheduler.get_last_lr()[0]:.2g}, patience: {early_stopper.patience}",end="")

        
        # train_loss = running_loss / len(train_dataloader)
        train_loss = val_epoch(model, train_dataloader, criterion)
        val_loss = val_epoch(model, val_dataloader, criterion)
        lr_scheduler.step(train_loss)
        early_stopper.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(lr_scheduler.get_last_lr())
        
        # checkpointing
        if checkpointing:
            if train_losses[-1] == min(train_losses):
                checkpoint = model.state_dict().copy()
        
        print(f"\rEpoch [{epoch+1}], Minibatch [{i+1}/{len(train_dataloader)}] Train: {0 if len(train_losses) == 0 else train_losses[-1]:.4g} Val: {0 if len(val_losses) == 0 else val_losses[-1]:.4g} lr: {lr_scheduler.get_last_lr()[0]:.2g}, patience: {early_stopper.patience}",end="")
        epoch += 1

    if checkpointing:
        return train_losses, val_losses, lrs, checkpoint
    else:
        return train_losses, val_losses, lrs

def train_model_config(lr, batch_size, layer_size, train_dataset, val_dataloader, lr_scheduler=None, early_stopper=None):
    if early_stopper is None:
        early_stopper = EarlyStopper(15)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size, 
        shuffle=True, 
        generator=torch.Generator(device=device), 
        num_workers=2,
        prefetch_factor=2,
        drop_last=True,
        persistent_workers=True)

    model = Exogenous_RkRNN(6, 2, layer_size)
    model.apply(init_weights)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    if lr_scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=7)
    else:
        lr_scheduler = lr_scheduler(optimizer)

    train_loss, val_loss, lrs, checkpoint = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, lr_scheduler, early_stopper)
    model = model.load_state_dict(checkpoint)
    return train_loss, val_loss, lrs, model

def hyperparameter_model_filename(params):
    return f"model_lr_{params['learning rate']}_bs_{params['batch size']}_ls_{params['layer size']}_ne_{params['number of epochs']}.pt"

def sefk_updating_trial(
    drifting_condition, 
    thresh=None, 
    quantile_thresh=None, 
    log_header=None, 
    lr=1e-5, 
    q=0.1, 
    p0=100, 
    logger=None
    ):
    if logger is None:
        logger = h5_logger(os.path.join(results_dir, "data.h5"))
    if log_header is None:
        if thresh is not None:
            log_header = f"{drifting_condition}/updating/t-{thresh:.2f}/"
        elif quantile_thresh is not None:
            log_header = f"{drifting_condition}/updating/q-{quantile_thresh:.2f}/"
        else:
            log_header = f"{drifting_condition}/updating/full/"
    print(log_header)
    if not logger.check_key(log_header+"xp"):
        drifting_data = get_glucose_data(logger, f"{drifting_condition}/data/")
        drifting_dataset = GlucoseDataset(drifting_data,
            prediction_horizon=60,
            input_horizon=1,
            step_size=DATA_STEP_SIZE,
            databounds=[0, 1],
            Xscaler=Xscaler,
            Uscaler=Uscaler
        )
        drifting_dataloader = torch.utils.data.DataLoader(drifting_dataset, batch_size=1, shuffle=False, generator=torch.Generator(device=device))

        model = torch.load(os.path.join(results_dir, "model-PC.pt"), map_location=device)
        optimizer = GEKF(model.parameters(), lr=lr, q=q, p0=p0)
        criterion = torch.nn.MSELoss()

        logger.log_attribute(log_header+"parameters/lr", np.array([lr]))
        logger.log_attribute(log_header+"parameters/q", np.array([q]))
        logger.log_attribute(log_header+"parameters/p0", np.array([p0]))

        start_time = time.time()

        model.train()
        pbar = tqdm(drifting_dataloader, desc=log_header)
        for data in pbar:
            optimizer.zero_grad()
            X0, U, X = data["X0"], data["U"], data["X"]
            with torch.no_grad():
                j = get_jacobian(model, (X0, U))
            X_pred = model(X0, U)
            innovation = X - X_pred
            loss = criterion(X_pred, X)
            loss.backward()
            grads = get_parameter_gradient_vector(model)
            mask = mask_fn(grads, thresh=thresh, quantile_thresh=quantile_thresh)
            optimizer.step(innovation, j, mask)
            pbar.set_description(f"{log_header} Loss: {np.mean(loss.item()):.4f}")
            
            logger.log_dict({
                log_header+"x0": X0.detach().cpu().numpy(),
                log_header+"u": U.detach().cpu().numpy(),
                log_header+"x": X.detach().cpu().numpy(),
                log_header+"xp": X_pred.detach().cpu().numpy(),
                log_header+"innovation": innovation.detach().cpu().numpy(),
                log_header+"loss": np.mean(loss.item()),
                log_header+"mask": mask.detach().cpu().numpy(),
                log_header+"weights": optimizer._get_flat_params().detach().cpu().numpy(),
                log_header+"time": np.array([time.time() - start_time])
            })
            


def plot_prediction(idx, x0, u, x, xp):
    t = np.arange(60)+1
    scatter_kwargs = dict(c='grey', s=5, label='Initial')
    true_kwargs = dict(c='b')
    pred_kwargs = dict(c='r', linestyle='dashed')
    u0_kwargs = dict(c='g', label='CV')
    u1_kwargs = dict(c='m', label='Bolus')
    bar_width = 0.4
    multipliers = np.array([0,1], dtype=int)

    with plt.rc_context({'lines.linewidth': 1,
        "axes.labelsize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5}):
        fig, ax = plt.subplots(4, 1, figsize=(4, 3), sharex=True, dpi=1000)
        # G
        ax[0].scatter(0, x0[idx, 0, 0], **scatter_kwargs)
        true_line = ax[0].plot(t, x[idx, :, 0], **true_kwargs)
        predicted_line = ax[0].plot(t, xp[idx, :, 0], **pred_kwargs)
        ax[0].set_ylabel('Blood Glucose\n(mg/dL)')
        # IX  # TODO: label each line
        ax[1].scatter(0, x0[idx, 0, 1], **scatter_kwargs)
        ax1_X = ax[1].plot(t, x[idx, :, 1], label="X", **true_kwargs)
        ax[1].plot(t, xp[idx, :, 1], **pred_kwargs)
        ax[1].scatter(0, x0[idx, 0, 2], **scatter_kwargs)
        ax1_I = ax[1].plot(t, x[idx, :, 2], label="I", c="tab:purple")
        ax[1].plot(t, xp[idx, :, 2], **pred_kwargs)
        ax[1].set_ylabel('Insulin\n(μU/mL)')
        # Q1, Q2, G_gut  TODO: label each line
        ax[2].scatter(0, x0[idx, 0, 3], **scatter_kwargs)
        ax[2].plot(t, x[idx, :, 3], label="q1", **true_kwargs)
        ax[2].plot(t, xp[idx, :, 3], **pred_kwargs)
        ax[2].scatter(0, x0[idx, 0, 4], **scatter_kwargs)
        ax[2].plot(t, x[idx, :, 4], label="q2", c="tab:purple")
        ax[2].plot(t, xp[idx, :, 4], **pred_kwargs)
        ax[2].scatter(0, x0[idx, 0, 5], **scatter_kwargs)
        ax[2].plot(t, x[idx, :, 5], label=r"G$_{gut}$", c="tab:pink")
        ax[2].plot(t, xp[idx, :, 5], **pred_kwargs)
        ax[2].set_ylabel('Carbs (g)')
        # ax[2].set_ylim(ax[2].get_ylim()[0]-5, ax[2].get_ylim()[1]+5)
        # ax[2].yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        # ax[2].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: int(x)))
        # ax[2].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        # U[0]: Insulin infusion
        # ax[3].step(t-1, u[idx, :, 0], where="post", **u0_kwargs)
        ax[3].bar(t-1, u[idx, :, 0], width=bar_width, color="tab:blue", label="u")
        ax[3].set_ylabel('Insulin Delivered\n(U/min)', color="tab:blue")
        ax[3].set_ylim(0, 0.1)
        # ax[3].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.02))
        # D[0]: Meal disturbance
        # ax3 = ax[2].twinx()
        ax3 = ax[3].twinx()
        ax3.bar(t-1+bar_width, u[idx, :, 1], width=bar_width, color="g", label="D")
        ax3.set_ylim(0, 150)
        ax3.set_ylabel('D (g)', color="g")
        # ax3.set_ylim(-2, 150)
        ax3.spines["right"].set_color("g")
        ax[3].spines["right"].set_color("g")
        ax3.spines["left"].set_color("tab:blue")
        ax[3].spines["left"].set_color("tab:blue")
        # ax[3].yaxis.set_major_locator(mpl.ticker.MultipleLocator(40))
        ax[3].set_xlim(-1, 62)
        ax[3].set_xlabel('Time (min)')
        ax[3].xaxis.set_label_coords(0.33, -.25)
        # ax[3].xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        # ax[3].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: int(x)))
        # ax[3].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))

        ax1_lines = ax[1].get_lines()
        labelLine(ax1_lines[0], 20, outline_width=3)
        labelLine(ax1_lines[2], 40, outline_width=3)
        ax2_lines = ax[2].get_lines()
        labelLine(ax2_lines[0], 15, outline_width=3)
        labelLine(ax2_lines[2], 30, outline_width=3)
        labelLine(ax2_lines[4], 45, outline_width=3)
        
        ax[3].legend([true_line[0], predicted_line[0]], ["True", "Predicted"], loc="upper center", bbox_to_anchor=(0.7, -0.2), ncol=2, fontsize=5)
    return fig, ax


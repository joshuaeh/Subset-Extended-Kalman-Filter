""""""
##########
# TODO ###
##########
# checkpoint last model -> trainable class, also set frequency to just last or infrequently
# get the model to save the weights of the best, delete others
# same summary stats

#############################
########## Imports ##########
#############################
import os
import shutil

from dtaidistance import dtw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from scipy.integrate import solve_ivp
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm

from sekf.modeling import init_weights
from sekf.optimizers import maskedAdam, SEKF

########## Config ##########

# Randomly sample initial conditions for training and testing
N_TRAIN = 100_000
N_VALIDATION = 10_000
N_TEST = 10_000
N_TRANSFER = 1000
N_TRANSFER_VALIDATION = 1000
N_TRANSFER_TOTAL = 100_000

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL0_FILENAME = "modelv0.pth"
MODEL0_PATH = os.path.join(CASE_DIR, MODEL0_FILENAME)
MODEL0_WEIGHTS_FILENAME = "modelv0_weights.pth"
MODEL0_WEIGHTS_PATH = os.path.join(CASE_DIR, MODEL0_WEIGHTS_FILENAME)
MODEL_FILENAME = "model.pth"
DATA_DIR = os.path.join(CASE_DIR, "data")
TRAINING_DATA_FILENAME = "training_data.npz"
TRAINING_METRICS_FILENAME = "training_metrics.npz"

N_CPUS = 20

ray.init(ignore_reinit_error=True)
assert ray.is_initialized()



# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 20)
    def forward(self, x):
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        x = self.fc3(x)
        x = nn.Sigmoid()(x)
        x = self.fc4(x)
        return x

class DampedSpringTrainer(tune.Trainable):
    def setup(self, config):
        self.config = config
        
        self.train_x = config["train_x"]
        self.train_y = config["train_y"]
        self.val_x = config["val_x"]
        self.val_y = config["val_y"]
        self.test_x = config["test_x"]
        self.test_y = config["test_y"]
        
        self._initialize_model(config)
        self.optimizer = maskedAdam(self.model.parameters(), lr=config["lr"], mask_fn_quantile_thresh=config["mask_fn_quantile_thresh"])
        self.loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=20, factor=0.5)
        self.train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.train_x, self.train_y),
            batch_size=config["batch_size"],
            shuffle=True,
        )
        self.initial_weights = self.optimizer._get_flat_params().detach().numpy()
        
    def _initialize_model(self, config):
        self.model = MLP()
        # load pre-trained weights
        weights_dict = torch.load(MODEL0_WEIGHTS_PATH, weights_only=True)
        self.model.load_state_dict(weights_dict)
        

    def step(self, train_x, train_y, val_x, val_y, test_x, test_y):
        self.model.train()
        for x_batch, y_batch in self.train_dataloader:
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.loss_fn(y_pred, y_batch)
            loss.backward()
            self.optimizer.masked_step()

        with torch.no_grad():
            epoch_loss = self.loss_fn(self.model(train_x), train_y).item()
            validation_loss = self.loss_fn(self.model(val_x), val_y).item()
            test_loss = self.loss_fn(self.model(test_x), test_y).item()
        self.scheduler.step(epoch_loss)
        # Log the training loss
        metrics = {
            "train_loss": epoch_loss,
            "val_loss": validation_loss,
            "test_loss": test_loss,
            "cosine_distance_weights": cosine_distance(self.initial_weights.reshape(1, -1), self.optimizer._get_flat_params().detach().numpy().reshape(1, -1)),
        }
        # tune.report(metrics)
        # checkpoint
        return metrics

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.model.load_state_dict(torch.load(checkpoint_path))
        return 


# Mass-Spring-Damper system simulation
def mass_spring_damper(t, state, m, c, k, u):
    x, x_dot = state
    x_ddot = (u - c*x_dot - k*x) / m
    return [x_dot, x_ddot]

def sim_mass_spring_damper(x0=0.0, x_dot0=0.0, m=1.0, c=0.5, k=1.0, u=0.0, t_end=20.0, t_eval=None):
    initial_state = [x0, x_dot0]  # Initial position and velocity
    params = (m, c, k, u)
    # Integrate the ODE
    res = solve_ivp(mass_spring_damper, [0,t_end], initial_state, args=params, t_eval=t_eval)
    return res.t, res.y

def generate_dataset(x0_samples, x_dot0_samples, **kwargs):
    X = []
    Y = []
    for x0, x_dot0 in tqdm(zip(x0_samples, x_dot0_samples)):
        t_sim, states_sim = sim_mass_spring_damper(x0=x0, x_dot0=x_dot0, **kwargs)
        X.append([x0, x_dot0])
        Y.append(states_sim[0])
    return np.array(X), np.array(Y)

## cosine similarity
# use einsum to compute Y_train @ Y_transfer.T / ||Y_train|| * ||Y_transfer||
def cosine_similarity(a, b):
    """computes row-wise cosine similarity."""
    assert a.ndim == b.ndim
    if a.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if a.ndim == 2:
        return np.einsum('ij,ik->i', a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

def cosine_distance(a, b):
    """computes row-wise cosine distance."""
    return (1 - cosine_similarity(a, b)) / 2

def get_transfer_data(k, v, data_dir=os.path.join(DATA_DIR, "transfer")):
    scenario_filename = f"{k}({v}).npz"
    if not os.path.exists(os.path.join(data_dir, scenario_filename)):
        return False
    data = np.load(os.path.join(data_dir, scenario_filename))
    data_x = np.load(os.path.join(data_dir, "X.npz"))["X"]
    data_y = data["Y"]
    return data_x, data_y

def train_val_test_split(x,y, n_train=None, n_validation=None, n_test=None, p_train=None, p_validation=None, p_test=None, tensor_convert=False):
    """Splits the data into train, validation and test sets."""
    assert len(x) == len(y), "x and y must have the same length"
    # give either n_train, n_validation or n_test, or p_train, p_validation, p_test
    if p_train is not None:
        assert p_train + p_validation + p_test == 1, "p_train, p_validation and p_test must sum to 1"
        assert all((n_train is None, n_validation is None, n_test is None)), "Either use p_train, p_validation, p_test or n_train, n_validation, n_test"
        n_train = int(p_train * len(x))
        n_validation = int(p_validation * len(x))
        n_test = len(x) - n_train - n_validation
    if n_train is None:
        n_train = int(0.8 * len(x))
    if n_validation is None:
        n_validation = int(0.1 * len(x))
    if n_test is None:
        n_test = len(x) - n_train - n_validation

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_validation = x[n_train:n_train+n_validation]
    y_validation = y[n_train:n_train+n_validation]
    x_test = x[n_train+n_validation:n_train+n_validation+n_test]
    y_test = y[n_train+n_validation:n_train+n_validation+n_test]
    if tensor_convert:
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_validation = torch.tensor(x_validation, dtype=torch.float32)
        y_validation = torch.tensor(y_validation, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    return x_train, y_train, x_validation, y_validation, x_test, y_test



########## Training Data ##########

if not os.path.exists(os.path.join(DATA_DIR, TRAINING_DATA_FILENAME)):
    x0_train = np.random.uniform(-5, 5, N_TRAIN)
    x_dot0_train = np.random.uniform(-5, 5, N_TRAIN)
    x0_validation = np.random.uniform(-5, 5, N_VALIDATION)
    x_dot0_validation = np.random.uniform(-5, 5, N_VALIDATION)
    x0_test = np.random.uniform(-5, 5, N_TEST)
    x_dot0_test = np.random.uniform(-5, 5, N_TEST)

    os.makedirs("data", exist_ok=True)
    
    X_train, Y_train = generate_dataset(x0_train, x_dot0_train, t_eval=np.arange(1, 21, 1))
    X_validation, Y_validation = generate_dataset(x0_validation, x_dot0_validation, t_eval=np.arange(1, 21, 1))
    X_test, Y_test = generate_dataset(x0_test, x_dot0_test, t_eval=np.arange(1, 21, 1))
    np.savez(os.path.join(DATA_DIR, TRAINING_DATA_FILENAME), X_train=X_train.astype(np.float32), Y_train=Y_train.astype(np.float32), X_validation=X_validation.astype(np.float32), Y_validation=Y_validation.astype(np.float32), X_test=X_test.astype(np.float32), Y_test=Y_test.astype(np.float32))

t_query = np.arange(1, 21, 1)  # t = 1, 2, ..., 20    
training_data = np.load(os.path.join(DATA_DIR, TRAINING_DATA_FILENAME))

X_train, Y_train = training_data['X_train'], training_data['Y_train']
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_validation, Y_validation = training_data['X_validation'], training_data['Y_validation']
X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
Y_validation_tensor = torch.tensor(Y_validation, dtype=torch.float32)
X_test, Y_test = training_data['X_test'], training_data['Y_test']
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)


if not os.path.exists(MODEL0_PATH):
    model = MLP()
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    loss_fn = nn.MSELoss()

    # Training loop
    n_epochs = 1000
    batch_size = 128

    train_loss = np.empty(n_epochs)
    validation_loss = np.empty(n_epochs)
    learning_rate = np.empty(n_epochs)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0.0
        for i in range(0, X_train_tensor.size(0), batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X_train_tensor[idx]
            y_batch = Y_train_tensor[idx]
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        # store/output metrics
        model.eval()
        with torch.no_grad():
            train_loss[epoch] = loss_fn(model(X_train_tensor), Y_train_tensor).item()
            scheduler.step(train_loss[epoch])
            validation_loss[epoch] = loss_fn(model(X_validation_tensor), Y_validation_tensor).item()
            learning_rate[epoch] = scheduler.optimizer.param_groups[0]['lr']
            if ((epoch+1) % 100 == 0) or (epoch < 10):
                print(f"Epoch {str(epoch+1).zfill(4)}/{n_epochs}, Train Loss: {train_loss[epoch]:.6e}, Validation Loss: {validation_loss[epoch]:.6e}, Learning Rate: {learning_rate[epoch]:.6e}")

    torch.save(model, MODEL0_PATH)
    torch.save(model.state_dict(), MODEL0_WEIGHTS_PATH)
    np.savez(os.path.join("data", TRAINING_METRICS_FILENAME), train_loss=train_loss, validation_loss=validation_loss, learning_rate=learning_rate)
else:
    model = torch.load(MODEL0_PATH, weights_only=False)
    train_loss, validation_loss, learning_rate = np.load(os.path.join("data", TRAINING_METRICS_FILENAME))['train_loss'], \
        np.load(os.path.join("data", TRAINING_METRICS_FILENAME))['validation_loss'], \
        np.load(os.path.join("data", TRAINING_METRICS_FILENAME))['learning_rate']

# fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# ax[0].plot(train_loss, label='Train Loss')
# ax[0].plot(validation_loss, label='Validation Loss')
# ax[0].set_yscale('log')
# ax[0].set_ylabel('Loss (log scale)')
# ax[0].legend()
# ax[1].plot(learning_rate, label='Learning Rate')
# ax[1].set_xlabel('Epoch')
# ax[1].set_ylabel('Learning Rate')
# ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.show()

# N_validation_examples = 5
# examples_indices = np.random.choice(X_validation_tensor.size(0), N_validation_examples, replace=False)

# t_plot = np.arange(1, 21)  # Time for plotting
# fig, ax = plt.subplots(N_validation_examples, 1, figsize=(10, 2 * N_validation_examples), sharex=True)
# for i, idx in enumerate(examples_indices):
#     x0, x_dot0 = X_validation_tensor[idx]
#     t_sim, states_sim = sim_mass_spring_damper(x0=x0.item(), x_dot0=x_dot0.item(), t_eval=np.linspace(0,20,200))
#     y_pred = model(X_validation_tensor[idx:idx+1]).detach().numpy().flatten()
#     # initial position
#     ax[i].plot(0, x0.item(), "bo", label="Initial Position")
#     # arrow showing initial velocity with length proportional to x_dot0 and number to the left showing its value
#     ax[i].arrow(0, x0.item(), 0, x_dot0.item(), length_includes_head=True, head_width=0.25, head_length=0.25, fc='red', ec='red')
#     ax[i].text(-0.1, x0.item() + x_dot0.item() / 2, f'{x_dot0.item():.2f}', color='red', fontsize=10, ha='right')
#     ax[i].plot(t_sim, states_sim[0], label='True Position', color='blue')
#     ax[i].plot(t_plot, y_pred, label='Predicted Position', marker="o", color='orange', linestyle='none')
#     ax[i].hlines(0, 0, 21, color="k", linestyle='--', linewidth=0.5)
#     ax[i].set_title(fr'Initial Conditions: $x_0$={x0.item():.2f}, $\dot{{x}}_0$={x_dot0.item():.2f}', pad=-14)
#     ax[i].set_ylabel('Position (x)')
#     # ax[i].legend()
# ax[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3)
# ax[i].set_xlabel('Time (t)')
# fig.tight_layout()
# plt.show()

# Transfer Learning
scenarios = [
    {"m":1.1},
    {"m":0.9},
    {"c":0.55},
    {"c":0.45},
    {"k":1.1},
    {"k":0.9},
    {"u":1.1},
    {"u":0.9},
    ]
# for kw in scenarios:
#     t_plot = np.arange(1, 21)  # Time for plotting
#     fig, ax = plt.subplots(N_validation_examples, 1, figsize=(10, 2 * N_validation_examples), sharex=True)
#     for i, idx in enumerate(examples_indices):
#         for k,v in kw.items():
#             x0, x_dot0 = X_validation_tensor[idx]
#             t_sim, states_sim = sim_mass_spring_damper(x0=x0.item(), x_dot0=x_dot0.item(), t_eval=np.linspace(0, 20, 200))
#             t_sim_transfer, states_sim_transfer = sim_mass_spring_damper(x0=x0.item(), x_dot0=x_dot0.item(), t_eval=np.linspace(0, 20, 200), **kw)
#             y_pred = model(X_validation_tensor[idx:idx+1]).detach().numpy().flatten()
#             # initial position
#             ax[i].plot(0, x0.item(), "bo", label='Initial Position')
#             # arrow showing initial velocity with length proportional to x_dot0 and number to the left showing its value
#             ax[i].arrow(0, x0.item(), 0, x_dot0.item(), length_includes_head=True, head_width=0.25, head_length=0.25, fc='red', ec='red')
#             ax[i].text(-0.1, x0.item() + x_dot0.item() / 2, f'{x_dot0.item():.2f}', color='red', fontsize=10, ha='right')
#             ax[i].plot(t_sim, states_sim[0], label='Original System', color='blue')
#             ax[i].plot(t_sim_transfer, states_sim_transfer[0], label=f"{k}={v}", color='blue', linestyle='--')
#             ax[i].plot(t_plot, y_pred, label='Predicted Position', marker="o", color='orange', linestyle='none')
#             ax[i].hlines(0, 0, 21, color="k", linestyle='--', linewidth=0.5)
#             ax[i].set_title(fr'Initial Conditions: $x_0$={x0.item():.2f}, $\dot{{x}}_0$={x_dot0.item():.2f}', pad=-14)
#             ax[i].set_ylabel('Position (x)')
#         # ax[i].legend()
#     ax[i].set_xlabel('Time (t)')
#     ax[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=4)
#     for k, v in kw.items():
#         fig.savefig(f"figures/transfer_learning_{k}({v}).png", bbox_inches='tight')
#         fig.savefig(f"figures/transfer_learning_{k}({v}).eps", bbox_inches='tight')
# fig.text(0.5, 0.95, 'Transfer Learning: Original vs Transferred System', ha='center', va='top', fontsize=16)


### Generate transfer learning data
if not os.path.exists(os.path.join(DATA_DIR, "transfer", "X.npy")):
    rng = np.random.default_rng(42)
    X_transfer = rng.uniform(-5, 5, (100000, 2))
    np.savez(os.path.join(DATA_DIR, "transfer", "X"), X=X_transfer)
else:
    X_transfer = np.load(os.path.join(DATA_DIR, "transfer", "X.npy"))["X"]

for kw in scenarios:
    for k,v in kw.items():
        pass
    if not get_transfer_data(k, v):
        Y_transfer = []
        for x in tqdm(X_transfer.reshape(-1,2)):
            x0, x_dot0 = x
            t_sim, states_sim = sim_mass_spring_damper(x0=x0, x_dot0=x_dot0, t_eval=np.arange(1, 21, 1), **kw)
            Y_transfer.append(states_sim[0])
        Y_transfer = np.array(Y_transfer, dtype=np.float32)
        scenario_filename = f"{k}({v}).npz"
        np.savez(os.path.join("data", "transfer", scenario_filename), Y=Y_transfer)

# if not os.path.exists("modelv0_dict.pth"):
#     model = torch.load("modelv0.pth", weights_only=False)
#     model_dict = model.state_dict()
#     torch.save(model_dict, "modelv0_dict.pth")


### Ray tune hyperparameters
for kw in scenarios:
    k,v = list(kw.items())[0]
    x, y = get_transfer_data(k, v)
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x, y, n_train=N_TRANSFER, n_validation=N_TRANSFER_VALIDATION, n_test=N_TRANSFER_TOTAL - N_TRANSFER - N_TRANSFER_VALIDATION, tensor_convert=True)

    config = {
        "lr": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128, 256, 500, 1000]),
        "max_epochs": 1000,
        "mask_fn_quantile_thresh": tune.uniform(0.0, 1.0),
        "num_trials": 200,
        # "train_x": x_train,
        # "train_y": y_train,
        # "val_x": x_val,
        # "val_y": y_val,
        # "test_x": x_test,
        # "test_y": y_test,
    }

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config["max_epochs"],
        grace_period=50,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                DampedSpringTrainer,
                train_x=x_train,
                train_y=y_train,
                val_x=x_val,
                val_y=y_val,
                test_x=x_test,
                test_y=y_test,
            ),
                
            resources={"cpu": N_CPUS},
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            max_concurrent_trials=N_CPUS,
            num_samples=config["num_trials"],
            reuse_actors=True
        ),
        param_space=config,
        run_config = tune.RunConfig(
            name=f"dampedSpring_{k}({v})",
            # storage_path=r"C:\Users\jhamm\Desktop\SEKF\transfer\dampedSpring\data\ray_results",
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True
            ),


        )
    )
    results = tuner.fit()

    best_result = results.get_best_result("val_loss", "min")

    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics}")
    metrics_df = results.get_dataframe()
    metrics_df.to_csv(os.path.join(DATA_DIR, "transfer", f"{k}({v})_allTrials_metrics.csv"))
    best_result_df = best_result.metrics_dataframe
    best_result_df.to_csv(os.path.join(DATA_DIR, "transfer", f"{k}({v})_bestResult_metrics.csv"))
    print(f"{best_result.path=}")
    print(f"{best_result.checkpoint=}")
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        model_path = os.path.join(checkpoint_dir, MODEL_FILENAME)
        target_path = os.path.join(DATA_DIR,"transfer", f"{k}({v})_model_weights.pth")
        shutil.move(model_path, target_path)

# delete /tmp/ray_results/dampedSpring_{k}({v})


# model = initialize_model()
# epoch_loss = loss_fn(model(x_train), y_train).item()
# validation_loss = loss_fn(model(x_val), y_val).item()

# print(x_train.shape, type(x_train))
# print(x_val.shape, type(x_val))
# print(x_test.shape, type(x_test))
# print(y_train.shape, type(y_train))
# print(y_val.shape, type(y_val))
# print(y_test.shape, type(y_test))
# print(f"Train Loss: {epoch_loss:.6e}, Validation Loss: {validation_loss:.6e}")
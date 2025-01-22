"""utilities to create, train, and manipulate models"""

import torch
from torch.func import functional_call, grad, vmap, jacrev, jacfwd
import numpy as np

from .utilities import bytes_dict

# TODO: Torch/numpy scaler class
# TODO: automatic device handling

# NN modules
class AbstractNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return 
    
    def _init_params(self):
        basic_weight_init(self)
    
    def functional_call(self, parameters, inputs):
        return torch.func.functional_call(self, parameters, inputs)
    
    def jacrev(self, inputs):
        return torch.func.jacrev(self.functional_call)(dict(self.named_parameters()), inputs)
    
    def jacrev_tensor(self, inputs):
        return format_jacobian(self.jacrev(inputs), dict(self.named_parameters()))
    
class AbstractRNN(AbstractNN):
    """Instead of each unit recurring, the entire network is recurrent. Outputs of NN are used as states at the next time.
    ARGS:
        inner_module (torch.nn.Module): should take N_states + N_inputs and return N_states
    """
    def __init__(self, inner_module):
        super().__init__()
        self.inner_module = inner_module
        return 
    
    def forward(self, X0, U):
        """Forward pass of the RNN
        ARGS:
            X0 (torch.Tensor): ([N_batch], 1, N_states) initial state
            U (torch.Tensor): ([N_batch], N_seq, N_inputs) input sequence
        RETURNS:
            X1 (torch.Tensor): ([N_batch], N_seq, N_states) next state
        """
        # determine if batched
        if X0.dim() == 2:
            assert U.dim() == 2, f"U must be 2D if X0 is 2D, got {U.dim()}"
            _is_batched = False
            X0 = X0.unsqueeze(0)
            U = U.unsqueeze(0)
        else:
            assert U.dim() == 3, f"U must be 3D if X0 is 3D, got {U.dim()}"
            _is_batched = True
        
        out = torch.empty(X0.shape[0], U.shape[1], X0.shape[2])
        
        x = X0
        for i in range(U.shape[1]):
            # concatenate the input and state [N_batch, 1, N_states + N_inputs]
            x_ = torch.cat([x, U[:,i].unsqueeze(1)], dim=-1)
            x = self.inner_module(x_)
            out[:,i] = x.squeeze(1)
        return out.squeeze(0) if not _is_batched else out

# RK4 integrator using PyTorch nn module
class RK4(torch.nn.Module):
    def __init__(self):
        super(RK4, self).__init__()
        return
    
    def forward(self, f, x, u, dt=1):
        k1 = f(x, u)
        k2 = f(x + 0.5*dt*k1, u)
        k3 = f(x + 0.5*dt*k2, u)
        k4 = f(x + dt*k3, u)
        return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

class VanillaRNN(AbstractNN):
    """ """
    # TODO: assumes 2 hidden layers each of hidden_size, could be generalized
    def __init__(self, state_dim, input_dim, hidden_size):
        super(VanillaRNN, self).__init__()
        self.linear1 = torch.nn.Linear(state_dim + input_dim, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, state_dim)
        return
    
    def forward(self, x0, u):
        """ARGS:
        x0: initial state (batch_size, context_length, input_dim)
        u: control input (batch_size, prediction_horizon, input_dim)
        RETURNS:
        x_pred: predicted states (batch_size, prediction_horizon, state_dim)
        """
        x_pred = torch.zeros(u.shape[0], u.shape[1], self.linear3.out_features)
        x = x0
        for i in range(u.shape[1]):
            x = torch.cat([x, u[:, i, :].unsqueeze(1)], dim=-1)
            x = torch.sigmoid(self.linear1(x))
            x = torch.sigmoid(self.linear2(x))
            x = self.linear3(x)
            x_pred[:, i, :] = x.squeeze(1)
        return x_pred

class Exogenous_MLP(AbstractNN):
    """ """
    def __init__(self, state_dim, input_dim, hidden_size):
        super(Exogenous_MLP, self).__init__()
        self.linear1a = torch.nn.Linear(input_dim, hidden_size)
        self.linear1b = torch.nn.Linear(state_dim, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, state_dim)
        return
    
    def forward(self, x, u):
        # xu = torch.cat([x, u], dim=-1)
        xu = torch.sigmoid(self.linear1a(u) + self.linear1b(x))
        xu = torch.sigmoid(self.linear2(xu))
        xu = self.linear3(xu)
        return xu

class Exogenous_RkRNN(AbstractNN):
    """ """
    # TODO: assumes 2 hidden layers each of hidden_size, could be generalized
    def __init__(self, state_dim, input_dim, hidden_size):
        super(Exogenous_RkRNN, self).__init__()
        self.mlp = Exogenous_MLP(state_dim, input_dim, hidden_size)
        self.rk4 = RK4()
        return
    
    def forward(self, x0, u):
        """ARGS:
        x0: initial state (batch_size, context_length, input_dim)
        u: control input (batch_size, prediction_horizon, input_dim)
        RETURNS:
        x_pred: predicted states (batch_size, prediction_horizon, state_dim)
        """
        x_pred = torch.zeros(u.shape[0], u.shape[1], x0.shape[-1])
        x = x0
        for i in range(u.shape[1]):
            # MLP rates are integrated using RK4
            x = self.rk4(self.mlp, x, u[:, i, :].unsqueeze(1))
            # prediction stored
            x_pred[:, i, :] = x.squeeze(1)
        return x_pred
    
class EarlyStopper:
    # TODO: add model checkpointing
    def __init__(self, patience=1, min_delta=0, true_on_stop=None):
        if (true_on_stop is None) or (true_on_stop == True):
            self.true_on_stop = True
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop = False

    def step(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
        return None
    
    def __call__(self):
        if self.true_on_stop:
            return self.counter >= self.patience
        else:
            return self.counter < self.patience

# functions for model interaction
def basic_weight_init(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            # xavier normal works well for symmetric nonlinearities
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    return

def init_load_model(model, model_kwargs, model_save_path, device):
    m = model(**model_kwargs)
    m.load_state_dict(torch.load(model_save_path, map_location=device))
    m.to(device)             

def get_parameter_info(model):
    total_elements = 0
    info_dict = {}
    for p in model.named_parameters():
        n_p = p[1].numel()
        info_dict[p[0]] = n_p
        total_elements += n_p
    return total_elements, info_dict

def get_parameters(model):
    """Get the parameters of a model as a dictionary"""
    parameters = {}
    for name, param in model.named_parameters():
        parameters[name] = param
    return parameters

def get_parameter_vector(model):
    """Get the parameters of a model as a 1D tensor"""
    return torch.cat([param.flatten() for name, param in model.named_parameters()]).squeeze()

def get_parameter_gradient_vector(model):
    """Get the gradients of the parameters of a model as a 1D tensor"""
    return torch.cat([param.grad.flatten() for name, param in model.named_parameters()]).squeeze()

def set_parameter_vector(model, parameter_vector):
    """Set the parameters of a model from a 1D tensor"""
    i = 0
    for name, param in model.named_parameters():
        n = param.numel()
        param.data = parameter_vector[i : i + n].view(param.shape)
        i += n

def set_parameter_gradient_vector(model, parameter_gradient_vector):
    """Set the gradients of the parameters of a model from a 1D tensor"""
    i = 0
    for name, param in model.named_parameters():
        n = param.numel()
        param.grad = parameter_gradient_vector[i : i + n].view(param.shape)
        i += n

def set_parameters(model, new_parameters):
    """Set the parameters of a model from a dictionary"""
    for name, param in new_parameters.items():
        model._parameters[name] = param

def get_selected_parameters(model, parameter_selection, n=None):
    """Get the selected parameters of a model as a 1D tensor"""
    for param_name, param_mask in parameter_selection.items():
        if param_name not in model._parameters:
            raise ValueError(f"Parameter {param_name} not found in model")
        if param_mask.shape != model._parameters[param_name].shape:
            raise ValueError(
                f"Parameter {param_name} has shape {model._parameters[param_name].shape} but mask has shape {param_mask.shape}"
            )
        with torch.no_grad():
            raise NotImplementedError("Not implemented yet")

def format_jacobian(jacobian_dict:dict, parameters:dict):
    """Format Jacobian dictionary into a single tensor (n_out, n_params)"""
    j_ = []
    for k,v in parameters.items():
        # reshaping this way ensures that batched/multistream are still organized per parameter
        numel = v.numel()
        j_.append(jacobian_dict[k].reshape(-1, numel))
    return torch.cat(j_, dim=1)
        
# TODO does this need to be wrapped in a torch.no_grad()?
# TODO batched jacobian calculation
@torch.no_grad()
def get_jacobian(model:torch.nn.Module,
    inputs:tuple,
    wrt="parameters",
    jac_mode=jacrev):
    # TODO: refactor for batched inputs
    """
    Get the jacobian of a model with respect to the parameters or inputs.
    ARGS:
        model: torch.nn.Module
        inputs: tuple
        wrt: str
        jac_mode: function
    RETURNS:
        jacobian: dict with keys as the parameter names and values as the jacobian for that parameter dy(row)/dx(col)"""
    # TODO: automatically select between jacrev and jacfwd based in n_inputs and n_outputs.
    #     forward will be faster for n_inputs << n_outputs, reverse when n_outputs << n_inputs
    assert jac_mode in [jacrev, jacfwd], f"jac_mode must be `jacrev` or `jacfwd`, got {jac_mode}"
    match wrt:
        case "parameters":
            f = lambda parameters, inputs: functional_call(model, parameters, inputs)
            jac_dict = jac_mode(f)(dict(model.named_parameters()), inputs)
        case "inputs":
            jac_dict = jac_mode(model)(inputs)
        case _:
            raise ValueError(f"wrt must be 'parameters' or 'inputs', got {wrt}")
    return format_jacobian(jac_dict, dict(model.named_parameters()))
    
def get_torch_model_size(model, units="mb"):
    """ """
    num_bytes = bytes_dict.get(units.upper(), 1024**2)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all = (param_size + buffer_size) / num_bytes
    print(f"model size: {size_all:.3f}{units.upper()}")
    return size_all

# model training
def MyCosineAnnealingWarmRestartsWithWarmup(optimizer,
    warmup_start_factor=0.1,
    warmup_end_factor=1.0,
    warmup_duration=10,
    T_0=10,
    T_mult=2,
    eta_min=0):
    """ """
    warmup_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=warmup_end_factor, total_iters=warmup_duration)
    cosine_lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr, cosine_lr], milestones=[warmup_duration])

def mask_fn(grads, thresh=None, quantile_thresh=None, template=None):
    if thresh is not None:
        return grads.abs() > thresh
    elif quantile_thresh is not None:
        return grads.abs() > torch.quantile(grads.abs(), quantile_thresh)
    elif template is not None:
        assert template.shape == grads.shape, f"Template shape {template.shape} must match grads shape {grads.shape}"
        return template
    else:
        return torch.ones_like(grads, dtype=torch.bool)
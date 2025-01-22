import torch

from .modeling import get_parameter_vector, mask_fn

class GenericOptimizer(torch.optim.Optimizer):
    def __init__(self, params, defaults):
        super(GenericOptimizer, self).__init__(params, defaults)
        return
    
    # @torch.no_grad()
    def _get_flat_params(self):
        return torch.cat([torch.cat([p.flatten() for p in param_group["params"]]) for param_group in self.param_groups])

    # @torch.no_grad()
    def _get_flat_grads(self):
        return torch.cat(
            [torch.cat([p.grad.flatten() for p in param_group["params"]]) for param_group in self.param_groups]
        )
        
class maskedSGD(torch.optim.SGD, GenericOptimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False, mask_kwargs=dict()):
        super(maskedSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.mask_kwargs = mask_kwargs
        return

    def masked_step(self, closure=None):
        """Only updates selected parameters defined by the mask or gradient threshold."""
        # either mask or grad_thresh must be provided
        mask = mask_fn(self._get_flat_grads(), **self.mask_kwargs)

        # get the pre-step parameters and gradients
        pre_step_params = self._get_flat_params()

        # normal step
        super(maskedSGD, self).step(closure=closure)

        # revert the parameters for parameters UNMASKED or BELOW GRADIENT THRESHOLD
        post_step_params = self._get_flat_params()
        post_step_params[~mask] = pre_step_params[~mask]
        self._set_flat_params(post_step_params)
        return (mask).sum().item()
    
class maskedAdam(torch.optim.Adam, GenericOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, mask_kwargs=dict()):
        super(maskedAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.mask_kwargs = mask_kwargs
        return

    def masked_step(self, closure=None):
        """Only updates selected parameters defined by the mask or gradient threshold."""
        # either mask or grad_thresh must be provided
        # assert mask is not None or grad_thresh is not None, "Either mask or grad_thresh must be provided."

        # get the pre-step parameters and gradients
        pre_step_params = self._get_flat_params()
        mask = mask_fn(self._get_flat_grads(), **self.mask_kwargs)

        # normal step
        super(maskedAdam, self).step(closure=closure)

        # revert the parameters for parameters UNMASKED or BELOW GRADIENT THRESHOLD
        post_step_params = self._get_flat_params()
        post_step_params[~mask] = pre_step_params[~mask]
        self._set_flat_params(post_step_params)
        return (mask).sum().item()
        

class SEKF(GenericOptimizer):
    """Global Extended Kalman Filter optimizer.
    ...
    IMPORTANT NOTE: This optimizer requires the gradient to be computed wrt the output of the model, NOT the loss.
    
    Generally lr and q are adjusted while leaving p0 at 100 for sigmoid activations and 1000 for linear activations.
    
    The process of doing the SEKF includes:
        1. Compute the innovation (e) and Jacobian (J) of the output wrt the parameters
        2. Compute the Loss and backpropagate to get the gradients (needed for mask fn)
        3. Take a step

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        q (float, optional): process noise (default: 1e-1) P&F recommend 0 for no noise up to 0.1. Generally annealed from large value to value on the order of 1e-6. This annealing helps convergence and by keeping non-zero helps avoid divergence of error covariance update
        p0 (float, optional): initial covariance matrix diagonal values (default: 1e-1)  P&F recommend 100 for sigmoid and 1,000 for linear activations
        mask_kwargs (dict, optional): kwargs for the mask function:
            thresh, quantile_thresh, template. If none are provided, all parameters are updated.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,  # initial learning rate
        q: float = 1e-1,  # process noise
        p0: float = 100,  # initial error covariance
        mask_kwargs: dict = dict()
    ):
        # validate arguments
        assert lr > 0, f"Learning rate must be positive, received {lr}"
        assert q >= 0, f"Process noise must be non-negative, received {q}"
        assert p0 > 0, f"Initial covariance must be positive, received {p0}"
        
        self.learning_rate = lr
        defaults = dict()
        super(SEKF, self).__init__(params, defaults)
        self._init_GEKF(p0, q)
        self.mask_kwargs = mask_kwargs
        return

    def _init_GEKF(self, p0, q):
        self.n_param_elements = sum(sum(p.numel() for p in param_group["params"]) for param_group in self.param_groups)
        self.P = torch.eye(self.n_param_elements) * p0
        self.Q = torch.eye(self.n_param_elements) * q
        self.W = self._get_flat_params()
        return
    
    @torch.no_grad()
    def step(self, e, J, verbose=False):
        """Performs a single optimization step.
        ARGS:
            e (torch.Tensor): (N_out) (N_out*N_stream) innovation, 
            J (torch.Tensor): (N_out, N_params) Jacobian dy/dW change in *output* wrt parameters
            mask (torch.Tensor): (N_params) mask of which parameters to update
            verbose: (bool) whether to return additional information
        Returns:
            None or dict: additional information if verbose=True
            
        Other Notes:
            P (N_param, N_param) is the error covariance matrix
            A (N_out, N_out) is the scaling matrix
            K (N_param, N_out) is the Kalman gain
            
            @torch.no_grad() is used to prevent the computation graph from being stored
        """
        # parse inputs
        e = e.reshape(-1, 1)  # column vector
        mask = self.mask_fn(self._get_flat_grads(), **self.mask_kwargs)
        
        A0 = torch.eye(J.shape[0]) / self.learning_rate + J[:,mask] @ self.P[mask][:,mask] @ J[:,mask].T
        A = torch.linalg.solve(A0, torch.eye(J.shape[0]))
        # A = torch.linalg.inv(
        #     # learning rate injects uniform addition to main diagonal
        #     torch.eye(J.shape[0]) / self.learning_rate  # JPJ^T is the covariance of the innovation
        #     + J[:,mask] @ self.P[mask][:,mask] @ J[:,mask].T
        # )
        K = self.P[mask][:,mask] @ J[:,mask].T @ A
        dW = (K @ e).reshape(-1)
        dP = -K @ J[:,mask] @ self.P[mask][:,mask] + self.Q[mask][:,mask]
        self.W[mask] += dW
        self.P[mask][:,mask] += dP
        self._set_flat_params(self.W)
        if verbose:
            return {
                "e": e,
                "J": J,
                "A": A,
                "K": K,
                "W": self.W,
                "P": self.P,
            }
        else:
            return
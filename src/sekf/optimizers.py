import torch


class basic_optimizer(torch.optim.Optimizer):
    """Base class for my optimizers that includes parameter access and setting utilities"""

    # @torch.no_grad()
    def _get_flat_params(self):
        return torch.cat([torch.cat([p.flatten() for p in param_group["params"]]) for param_group in self.param_groups])

    # @torch.no_grad()
    def _get_flat_grads(self):
        return torch.cat(
            [torch.cat([p.grad.flatten() for p in param_group["params"]]) for param_group in self.param_groups]
        )
        
    def _multidimensional_backprop(self, backpropagated_tensor):
        """Backpropagate through a multidimensional output tensor.
        Args:
            out: multidimensional output tensor
            retain_graph: whether to retain the computation graph
        Returns:
            torch.Tensor: (n_out, p_param) tensor of gradients
        """
        # TODO refactor for batched operations
        n_out = backpropagated_tensor.numel()
        params = self._get_flat_params()
        n_params = params.numel()
        partial_grads = torch.zeros(n_out, n_params)
        backpropagated_tensor = backpropagated_tensor.flatten()
        for i in range(n_out):
            self.zero_grad()
            partial_mask = torch.zeros(n_out)
            partial_mask[i] = 1
            torch.autograd.backward(backpropagated_tensor, grad_tensors=partial_mask, retain_graph=True, create_graph=True)
            partial_grads[i] = self._get_flat_grads()
        return partial_grads
    
    # TODO: refactor to include vectorized, funcitonal operations for multi-dimensional outputs

    # @torch.no_grad()
    def _set_flat_params(self, flat_params):
        idx = 0
        for param_group in self.param_groups:
            for p in param_group["params"]:
                numel = p.numel()
                p.data = flat_params[idx : idx + numel].reshape_as(p)
                idx += numel
        return


class maskedAdam(torch.optim.Adam, basic_optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, mask=None):
        super(maskedAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.mask = mask
        return

    def masked_step(self, mask=None, grad_thresh=None, grad_quantile=None, closure=None):
        """Only updates selected parameters defined by the mask or gradient threshold."""
        # either mask or grad_thresh must be provided
        # assert mask is not None or grad_thresh is not None, "Either mask or grad_thresh must be provided."

        # get the pre-step parameters and gradients
        pre_step_params = self._get_flat_params()
        if grad_thresh is not None:
            pre_step_grads = self._get_flat_grads()
            mask = pre_step_grads.abs() > grad_thresh
        
        elif grad_quantile is not None:
            pre_step_grads = self._get_flat_grads()
            quantile_value = pre_step_grads.abs().quantile(grad_quantile)
            mask = pre_step_grads.abs() > quantile_value
            
        elif mask is None:
            super(maskedAdam, self).step(closure=closure)
            return

        # normal step
        super(maskedAdam, self).step(closure=closure)

        # revert the parameters for parameters UNMASKED or BELOW GRADIENT THRESHOLD
        post_step_params = self._get_flat_params()
        post_step_params[~mask] = pre_step_params[~mask]
        self._set_flat_params(post_step_params)
        return (mask).sum().item()
    
class maskedSGD(torch.optim.SGD, basic_optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False, mask=None):
        super(maskedSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.mask = mask
        return

    def masked_step(self, mask=None, grad_thresh=None, closure=None):
        """Only updates selected parameters defined by the mask or gradient threshold."""
        # either mask or grad_thresh must be provided
        assert mask is not None or grad_thresh is not None, "Either mask or grad_thresh must be provided."

        # get the pre-step parameters and gradients
        pre_step_params = self._get_flat_params()
        if mask is None:
            pre_step_grads = self._get_flat_grads()
            mask = pre_step_grads.abs() > grad_thresh

        # normal step
        super(maskedSGD, self).step(closure=closure)

        # revert the parameters for parameters UNMASKED or BELOW GRADIENT THRESHOLD
        post_step_params = self._get_flat_params()
        post_step_params[~mask] = pre_step_params[~mask]
        self._set_flat_params(post_step_params)
        return (mask).sum().item()


class SEKF(basic_optimizer):
    """Subset Extended Kalman Filter optimizer.
    ...
    IMPORTANT NOTE: This optimizer requires the gradient to be computed wrt the output of the model, NOT the loss.
    
    Generally lr and q are adjusted while leaving p0 at 100 for sigmoid activations and 1000 for linear activations.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        q (float, optional): process noise (default: 1e-1) P&F recommend 0 for no noise up to 0.1. Generally annealed from large value to value on the order of 1e-6. This annealing helps convergence and by keeping non-zero helps avoid divergence of error covariance update
        p0 (float, optional): initial covariance matrix diagonal values (default: 1e-1)  P&F recommend 100 for sigmoid and 1,000 for linear activations
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,  # initial learning rate
        q: float = 1e-1,  # process noise
        p0: float = 100,  # initial error covariance
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.learning_rate = lr
        # defaults = dict(lr=lr)
        defaults = dict()
        super(SEKF, self).__init__(params, defaults)
        self._init_SEKF(p0, q)
        return

    def _init_SEKF(self, p0, q):
        self.n_param_elements = sum(sum(p.numel() for p in param_group["params"]) for param_group in self.param_groups)
        self.P = torch.eye(self.n_param_elements) * p0
        self.Q = torch.eye(self.n_param_elements) * q
        self.W = self._get_flat_params()
        return

    @torch.no_grad()
    def step(self, e, J, mask=None, verbose=False):
        """Performs a single optimization step.
        ARGS:
            e (torch.Tensor): (N_out) (N_out*N_stream) innovation, y_true - y_pred
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
        # parse inputs and initialize matricies
        if mask is None:
            mask = torch.ones(J.shape[1], dtype=torch.bool)
        else:
            assert mask.shape == (J.shape[1],), f"Mask must have the same number of elements as the number of parameters, received {mask.shape[0]} expected {J.shape[1]}"
        e = e.reshape(-1, 1)
        
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
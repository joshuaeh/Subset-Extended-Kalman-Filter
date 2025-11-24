import logging
import os

import numpy as np
import torch

from .modeling import mask_fn, get_jacobian

logger = logging.getLogger(__name__)


class PersistentSampler:
    """Sampler that maintains position across training intervals."""
    
    def __init__(self, dataset_size: int, seed: int = 42):
        self.dataset_size = dataset_size
        self.indices = np.arange(dataset_size)
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.indices)
        self.position = 0
        self.epochs_completed = 0
        self.samples_seen = 0
    
    def get_batch_indices(self, batch_size: int) -> np.ndarray:
        # Handle wrap-around
        if self.position + batch_size > self.dataset_size:
            self.rng.shuffle(self.indices)
            self.position = 0
            self.epochs_completed += 1
        
        batch_idx = self.indices[self.position:self.position + batch_size]
        self.position += batch_size
        self.samples_seen += batch_size
        return batch_idx
    
    @property
    def effective_epochs(self) -> float:
        return self.samples_seen / self.dataset_size
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "indices": self.indices.copy(),
            "rng_state": self.rng.bit_generator.state,
            "position": self.position,
            "epochs_completed": self.epochs_completed,
            "samples_seen": self.samples_seen,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        self.indices = state["indices"]
        self.rng.bit_generator.state = state["rng_state"]
        self.position = state["position"]
        self.epochs_completed = state["epochs_completed"]
        self.samples_seen = state["samples_seen"]

class basic_optimizer(torch.optim.Optimizer):
    """Base class for my optimizers that includes parameter access and setting utilities"""

    # TODO: add masking utilities to base optimizer
    # TODO: add state_dict serialization

    # @torch.no_grad()
    def _get_flat_params(self):
        return torch.cat(
            [
                torch.cat([p.flatten() for p in param_group["params"]])
                for param_group in self.param_groups
            ]
        )

    # @torch.no_grad()
    def _get_flat_grads(self):
        return torch.cat(
            [
                torch.cat([p.grad.flatten() for p in param_group["params"]])
                for param_group in self.param_groups
            ]
        )

    # def _multidimensional_backprop(self, backpropagated_tensor):
    #     """Backpropagate through a multidimensional output tensor.
    #     Args:
    #         out: multidimensional output tensor
    #         retain_graph: whether to retain the computation graph
    #     Returns:
    #         torch.Tensor: (n_out, p_param) tensor of gradients
    #     """
    #     # TODO refactor for batched operations
    #     n_out = backpropagated_tensor.numel()
    #     params = self._get_flat_params()
    #     n_params = params.numel()
    #     partial_grads = torch.zeros(n_out, n_params)
    #     backpropagated_tensor = backpropagated_tensor.flatten()
    #     for i in range(n_out):
    #         self.zero_grad()
    #         partial_mask = torch.zeros(n_out)
    #         partial_mask[i] = 1
    #         torch.autograd.backward(backpropagated_tensor, grad_tensors=partial_mask, retain_graph=True, create_graph=True)
    #         partial_grads[i] = self._get_flat_grads()
    #     return partial_grads

    # # TODO: refactor to include vectorized, functional operations for multi-dimensional outputs

    # @torch.no_grad()
    def _set_flat_params(self, flat_params):
        idx = 0
        for param_group in self.param_groups:
            for p in param_group["params"]:
                numel = p.numel()
                p.data = flat_params[idx : idx + numel].reshape_as(p)
                idx += numel
        return

    # TODO: add save optimizer/parameter state to file function to BasicOptimizer
    # logic: have a dictionary of saved parameters for each optimizer


class maskedAdam(torch.optim.Adam, basic_optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        mask=None,
        mask_fn_thresh=None,
        mask_fn_quantile_thresh=None,
    ):
        super(maskedAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.mask = mask
        self.mask_fn_thresh = mask_fn_thresh
        self.mask_fn_quantile_thresh = mask_fn_quantile_thresh
        return

    def masked_step(
        self, mask=None, grad_thresh=None, grad_quantile=None, closure=None
    ):
        """Only updates selected parameters defined by the mask or gradient threshold."""
        # either mask or grad_thresh must be provided
        # assert mask is not None or grad_thresh is not None, "Either mask or grad_thresh must be provided."

        # get the pre-step parameters and gradients
        pre_step_params = self._get_flat_params()
        # set mask.
        # mask arg has prescedence otherwise grad_thresh and grad_quantile
        if mask is None:
            mask = mask_fn(
                self._get_flat_grads(),
                thresh=self.mask_fn_thresh if grad_thresh is None else grad_thresh,
                quantile_thresh=self.mask_fn_quantile_thresh
                if grad_quantile is None
                else grad_quantile,
            )

        # normal step
        super(maskedAdam, self).step(closure=closure)

        # revert the parameters for parameters UNMASKED or BELOW GRADIENT THRESHOLD
        post_step_params = self._get_flat_params()
        post_step_params[~mask] = pre_step_params[~mask]
        self._set_flat_params(post_step_params)
        return (mask).sum().item()


class maskedSGD(torch.optim.SGD, basic_optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        mask=None,
    ):
        super(maskedSGD, self).__init__(
            params, lr, momentum, dampening, weight_decay, nesterov
        )
        self.mask = mask
        return

    def masked_step(self, mask=None, grad_thresh=None, closure=None):
        """Only updates selected parameters defined by the mask or gradient threshold."""
        # either mask or grad_thresh must be provided
        assert mask is not None or grad_thresh is not None, (
            "Either mask or grad_thresh must be provided."
        )

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

    # TODO: state_dict for saving and loading optimizer state (including P and Q matrices), lr, etc.
    def __init__(
        self,
        params,
        R: float = 1,  # Measurement noise
        p0: float = 100,  # initial error covariance
        q: float = 1e-1,  # process noise
        mask_fn_thresh=None,
        mask_fn_quantile_thresh=None,
    ):
        # todo check the inputs

        defaults = dict(
            R=R,
            q=q,
            p0=p0,
            mask_fn_thresh=mask_fn_thresh,
            mask_fn_quantile_thresh=mask_fn_quantile_thresh,
        )
        # defaults = dict()
        # super(SEKF, self).__init__(params, defaults)
        super().__init__(params, defaults)
        self._init_SEKF(R, p0, q, mask_fn_thresh, mask_fn_quantile_thresh)
        return

    def _init_SEKF(self, R, p0, q, mask_fn_thresh, mask_fn_quantile_thresh):
        """Initialize the SEKF P, Q matrices."""
        self.n_param_elements = sum(
            sum(p.numel() for p in param_group["params"])
            for param_group in self.param_groups
        )
        # R must be a scalar, a 1-element tensor, 1-D tensor with as many elements as NN outputs, or a 2-D tensor with shape (N_out, N_out)
        self.R = R
        self.P = torch.eye(self.n_param_elements) * p0
        self.Q = torch.eye(self.n_param_elements) * q
        self.W = self._get_flat_params()
        self.mask_fn_thresh = mask_fn_thresh
        self.mask_fn_quantile_thresh = mask_fn_quantile_thresh
        return

    @torch.no_grad()
    def step(self, e, J, R=None, mask=None):
        """Performs a single optimization step.
        ARGS:
            e (torch.Tensor): (N_out) (N_out*N_stream) innovation, y_true - y_pred
            J (torch.Tensor): (N_params, N_out)(N_params, N_out*N_stream) Jacobian dy/dW change in *output* wrt parameters
            R (torch.Tensor): (N_out, N_out) measurement noise covariance matrix, or None to use diagonal matrix with self.R on the diagonal
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
            if self.mask_fn_thresh is not None:
                mask = mask_fn(
                    self._get_flat_grads(),
                    thresh=self.mask_fn_thresh,
                )
            elif self.mask_fn_quantile_thresh is not None:
                mask = mask_fn(
                    self._get_flat_grads(),
                    quantile_thresh=self.mask_fn_quantile_thresh,
                )
            else:
                mask = torch.ones(J.shape[1], dtype=torch.bool)
        else:
            assert mask.shape == (J.shape[1],), (
                f"Mask must have the same number of elements as the number of parameters, received {mask.shape[0]} expected {J.shape[1]}"
            )
        if R is None:
            R = torch.eye(J.shape[0]) * self.R
        else:
            assert R.shape == (J.shape[0], J.shape[0]), (
                f"R must have shape ({J.shape[0]}, {J.shape[0]}), received {R.shape}"
            )
        e = e.reshape(-1, 1)  # (N_out * N_stream, 1) innovation column vector

        A0 = R + J[:, mask] @ self.P[mask][:, mask] @ J[:, mask].T
        A = torch.linalg.solve(A0, torch.eye(J.shape[0]))
        # A = torch.linalg.inv(
        #     # learning rate injects uniform addition to main diagonal
        #     torch.eye(J.shape[0]) / self.learning_rate  # JPJ^T is the covariance of the innovation
        #     + J[:,mask] @ self.P[mask][:,mask] @ J[:,mask].T
        # )
        self.K = self.P[mask][:, mask] @ J[:, mask].T @ A
        self.dW = (self.K @ e).reshape(-1)
        # self.dP = self.K @ J[:,mask] @ self.P[mask][:,mask]
        self.W[mask] = self.W[mask] + self.dW
        # self.P[mask][:,mask] = self.P[mask][:,mask] + self.dP
        # self.P[mask][:,mask] = (torch.eye(self.n_param_elements)[mask][:,mask] - self.K @ J[:,mask]) @ self.P[mask][:,mask] + self.dP
        # subset_P = (torch.eye(sum(mask)) - self.K @ J[:, mask]) @ self.P[mask][:, mask] @ (
        #     torch.eye(sum(mask)) - self.K @ J[:, mask]
        # ).T + self.Q[mask][:, mask]
        subset_P = (
            (torch.eye(sum(mask)) - self.K @ J[:, mask])
            @ self.P[mask][:, mask]
            @ (torch.eye(sum(mask)) - self.K @ J[:, mask]).T
            + self.K @ R @ self.K.T
            + self.Q[mask][:, mask]
        )
        mask_2d = mask.unsqueeze(0) & mask.unsqueeze(1)
        self.P.index_put_(torch.where(mask_2d), subset_P.flatten())
        self._set_flat_params(self.W)
        return

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        state = {
            "R": self.R,
            "P": self.P,
            "Q": self.Q,
            "W": self.W,
            "mask_fn_thresh": self.mask_fn_thresh,
            "mask_fn_quantile_thresh": self.mask_fn_quantile_thresh,
        }
        return state

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        self.R = state_dict["R"]
        self.P = state_dict["P"]
        self.Q = state_dict["Q"]
        self.W = state_dict["W"]
        self.mask_fn_thresh = state_dict["mask_fn_thresh"]
        self.mask_fn_quantile_thresh = state_dict["mask_fn_quantile_thresh"]
        self._set_flat_params(self.W)
        return

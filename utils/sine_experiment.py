from tqdm.notebook import tqdm, trange

from .__init__ import *
from . import vis
from . import modeling


def sine_process(x, noise_scale=0.15, rng=default_rng):
    y = np.sin(x) + rng.normal(0, noise_scale, size=x.shape)
    return y


def generate_data(
    n_points,
    n_test_points=None,
    domain_min=-np.pi,
    domain_max=np.pi,
    noise_scale=0.15,
    rng=default_rng,
    drift=0.0,
):
    x = rng.uniform(domain_min, domain_max, size=n_points)
    y = sine_process(x + drift, noise_scale, rng)
    if n_test_points is not None:
        x_test = rng.uniform(domain_min, domain_max, size=n_test_points)
        y_test = sine_process(x_test + drift, noise_scale, rng)
        return x, y, x_test, y_test
    else:
        return x, y


class MLP_Model(torch.nn.Module):
    def __init__(self, I, H1, O):
        super().__init__()
        self.linear1 = torch.nn.Linear(I, H1)
        self.linear2 = torch.nn.Linear(H1, O)
        self.sigmoid = torch.nn.Sigmoid()
        
        self.init_weights()

        # store attributes
        dim_parameters = 0
        for parameter_name, parameter in self.named_parameters():
            dim_parameters += parameter.numel()
        self.dim_parameters = dim_parameters

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data)

    def train_model(
        self,
        x,
        y,
        x_test,
        y_test,
        loss_fn,
        optimizer,
        epochs,
        batch_size,
        x_scaler,
        y_scaler,
        writer=None,
        print_frequency=1_000,
        graph_frequency=1_000,
        early_stopping_threshold=1e-5,
        model_masks=None,
        original_parameters=None,
        silent=False,
    ):
        train_losses = np.zeros(epochs)
        test_losses = np.zeros(epochs)
        for epoch in trange(epochs, disable=silent):
            # Training
            self.train()
            idx = np.arange(x.shape[0])
            default_rng.shuffle(idx)
            x = x[idx]
            y = y[idx]
            for batch_i in range(0, x.shape[0], batch_size):
                x_batch = x[batch_i : batch_i + batch_size]
                y_batch = y[batch_i : batch_i + batch_size]

                # Forward pass
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if model_masks is not None and original_parameters is not None:
                    with torch.no_grad():
                        for name, param in self.named_parameters():
                            restored_idx = ~model_masks[name]
                            param[restored_idx] = original_parameters[name][restored_idx]

            # Testing
            self.eval()
            with torch.no_grad():
                y_pred = self(x)
                train_losses[epoch] = loss_fn(y_pred, y).item()
                y_pred_test = self(x_test)
                test_losses[epoch] = loss_fn(y_pred_test, y_test).item()

            if epoch % print_frequency == 0:
                if silent is False:
                    tqdm.write(
                        f"Epoch {epoch} | Train Loss: {train_losses[epoch]:.4g} | Test Loss: {test_losses[epoch]:.4g}"
                    )

            if epoch % graph_frequency == 0:
                vis.graph_predictions(
                    self,
                    x,
                    x_test,
                    y,
                    y_test,
                    x_scaler,
                    y_scaler,
                    epoch=epoch,
                    writer=writer,
                )

            # Logging
            if writer is not None:
                writer.add_histogram("Model_Parameters/linear1_Weights", self.linear1.weight, epoch)
                writer.add_histogram("Model_Parameters/linear1_Biases", self.linear1.bias, epoch)
                writer.add_histogram("Model_Parameters/linear2_Weights", self.linear2.weight, epoch)
                writer.add_histogram("Model_Parameters/linear2_Biases", self.linear2.bias, epoch)

                writer.add_histogram(
                    "Model_Parameter_Gradients/linear1_Weights",
                    self.linear1.weight.grad,
                    epoch,
                )
                writer.add_histogram(
                    "Model_Parameter_Gradients/linear1_Biases",
                    self.linear2.weight.grad,
                    epoch,
                )
                writer.add_histogram(
                    "Model_Parameter_Gradients/linear2_Weights",
                    self.linear1.bias.grad,
                    epoch,
                )
                writer.add_histogram(
                    "Model_Parameter_Gradients/linear2_Biases",
                    self.linear2.bias.grad,
                    epoch,
                )

                writer.add_scalars(
                    "Losses",
                    {
                        "Loss/Train": train_losses[epoch],
                        "Loss/Test": test_losses[epoch],
                    },
                    epoch,
                )

            if train_losses[epoch] < early_stopping_threshold:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break

        return train_losses, test_losses


def init_log_storage(epochs, num_train_points, num_test_points):
    train_loss_mae = np.zeros(int(epochs + 1))
    train_loss_mse = np.zeros(int(epochs + 1))
    test_loss_mae = np.zeros(int(epochs + 1))
    test_loss_mse = np.zeros(int(epochs + 1))
    train_predictions = np.zeros((int(epochs + 1), num_train_points))
    test_predictions = np.zeros((int(epochs + 1), num_test_points))
    # return train_loss_mae, train_loss_mse, test_loss_mae, test_loss_mse, train_predictions, test_predictions
    return {
        "train_loss_mae": train_loss_mae,
        "train_loss_mse": train_loss_mse,
        "test_loss_mae": test_loss_mae,
        "test_loss_mse": test_loss_mse,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
    }


def log_epoch(num_epoch, train_pred, train_true, test_pred, test_true, log_storage):
    train_loss_mae = log_storage["train_loss_mae"]
    train_loss_mse = log_storage["train_loss_mse"]
    test_loss_mae = log_storage["test_loss_mae"]
    test_loss_mse = log_storage["test_loss_mse"]
    train_predictions = log_storage["train_predictions"]
    test_predictions = log_storage["test_predictions"]

    train_loss_mae[num_epoch] = np.mean(np.abs(train_pred - train_true))
    train_loss_mse[num_epoch] = np.mean((train_pred - train_true) ** 2)
    test_loss_mae[num_epoch] = np.mean(np.abs(test_pred - test_true))
    test_loss_mse[num_epoch] = np.mean((test_pred - test_true) ** 2)
    train_predictions[num_epoch] = train_pred
    test_predictions[num_epoch] = test_pred
    return

def adam_update_step(model, optimizer, loss_threshold, x_scaled, y_scaled, mask=None, grad_thresh=None, grad_quantile=None, retrain_epochs=50):
    y_pred = model(x_scaled)
    loss = torch.nn.functional.mse_loss(y_pred, y_scaled)
    loss.backward()
    gradients = optimizer._get_flat_grads()
    if loss.item() > loss_threshold:
        for i in range(retrain_epochs):
            optimizer.zero_grad()
            y_pred_retrain = model(x_scaled)
            loss_retrain = torch.nn.functional.mse_loss(y_pred_retrain, y_scaled)
            loss_retrain.backward()
            gradients = optimizer._get_flat_grads()
            mask = modeling.mask_fn(gradients, thresh=grad_thresh, quantile_thresh=grad_quantile)
            optimizer.masked_step(mask=mask)
    return y_pred, loss, gradients

def kf_update_step(model, optimizer, loss_threshold, x_scaled, y_scaled, mask=None, grad_thresh=None, grad_quantile=None):
    optimizer.zero_grad()
    y_pred = model(x_scaled)
    loss = torch.nn.functional.mse_loss(y_pred, y_scaled)
    loss.backward()
    grads = optimizer._get_flat_grads()
    mask = modeling.mask_fn(grads, thresh=grad_thresh, quantile_thresh=grad_quantile)
    innovation = y_scaled - y_pred
    j = modeling.get_jacobian(model, (x_scaled))
    optimizer.step(innovation, j, mask=mask)
    return y_pred, loss, grads



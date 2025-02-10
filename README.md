# Online Model Maintenance using the Subset Extended Kalman Filter

Code and recorded data for replication of results and figures.

- Jupyter notebook files for each of the presented case studies and the figures used in this work are in their respective `*.ipynb` files
- Scripts for each of the case studies, as well as customized optimizer and logger classes are available within the `utils` directory. For more information see the `README.md` file within the directory.
- Recorded results are saved in the `./results` directory. Due to the size of these files, they are available at [https://doi.org/10.18738/T8/VA0SQI](https://doi.org/10.18738/T8/VA0SQI).  

## Using the Subset Extended Kalman Filter

The Subset Extended Kalman Filter uses modified Extended Kalman Filter methods to perform recursive updates on a *subset* of model paramters that are identified using the gradient of the loss function with respect to model paramters. Operating on a subset of model parameters lowers the cost of the Kalman Filtering operations which increases exponentially with the number of model paramters updated.

The Subset Extended Kalman Filter is written in PyTorch and takes advantage of vectorization to calculate the Jacobian of each model output with respect to model parameters.

Example usage is as follows:

```python
import torch
from utils.optimizers import GEKF
from utils.modeling import get_jacobian, get_parameter_gradient_vector, mask_fn, init_weights

# create nn model
nn = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 1),
    )
nn.apply(init_weights)

opt = GEKF(nn.parameters(), lr=0.001)

# implement the desired loss function (MSE here as an example)
loss_fn = torch.nn.MSELoss()

# implement torch DataLoader to load data sequentially
# data_loader = 


for data_k in data_loader:
    nn_input, pred_true = data_k  # unpack input and true output from DataLoader

    opt.zero_grad()  # zero the tracked gradients in the optimizer
    xp = model(nn_input)  # calculate model prediction
    innovation = pred_true - xp  # calculate innovation: the difference between the prediction and true value
    loss = loss_fn(pred_true, xp)  # calculate the loss according to the desired loss function
    with torch.no_grad():
        j = get_jacobian(model, nn_input)  # use provided function to get the jacobian of outputs w.r.t. weights
    # backpropagate loss and calculate which parameters to update
    loss.backward()
    grads = get_parameter_gradient_vector(model)
    mask = mask_fn(grads)
    # perform SEKF optimization step
    opt.step(innovation, j, mask)

```

## Environment

Create a python 3.12 virtual environment or conda environment. Then, `pip install -r requirements.txt`. If issues surround the use of the `h5_logger` class of `h5py`, follow the instructions for the [h5py package installation](https://docs.h5py.org/en/stable/build.html#install) to make sure that the hdf5 binaries are added to your environment path.

## Citation

If you use this work, please consider citing it. A suggested citation:
```

```

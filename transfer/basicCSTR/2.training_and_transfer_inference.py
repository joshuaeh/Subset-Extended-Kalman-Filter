"""Evaluate the trained model on training/transfer data"""

# imports
from basicCSTR import *

# constants

# script
if __name__ == "__main__":
    # ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_RESULTS_DIR = RESULTS_DIR.joinpath("training")
    TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFER_DATA_PATH = DATA_DIR.joinpath("transfer_data.npz")
    TRANSFER_RESULTS_PATH = RESULTS_DIR.joinpath("transfer", "noMaintenance", "results.npz")
    TRANSFER_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    model_path = TRAINING_RESULTS_DIR.joinpath(MODEL_FILENAME)
    model = Exogenous_RkRNN(state_dim=3, input_dim=1, hidden_size=32)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # training data
    TRAINING_RESULTS_DATA = TRAINING_RESULTS_DIR.joinpath("results.npz")
    if ((not TRAINING_RESULTS_DATA.exists()) or (not TRANSFER_RESULTS_PATH.exists())):
        print("Getting training data")
        res = dict(np.load(TRAINING_DATA_PATH))
        for k, v in res.items():
            res[k] = v.astype(np.float64)
        
        data_len = len(res["Y"])
        val_test_index = data_len - 31 * 24 * 60  # last month test data
        train_val_index = val_test_index - 30 * 24 * 60  # prior month validation data
        
        data = {
            "train_y": res["Y"][:train_val_index],
            "train_u": res["U"][:train_val_index],
            "val_y": res["Y"][train_val_index:val_test_index],
            "val_u": res["U"][train_val_index:val_test_index],
            "test_y": res["Y"][val_test_index:],
            "test_u": res["U"][val_test_index:],
        }
        
        # fit scaler
        x_scaler = StandardScaler()
        x_scaler.fit(data["train_y"])
        u_scaler = StandardScaler()
        u_scaler.fit(data["train_u"].reshape(-1, 1))
        
    if not TRAINING_RESULTS_DATA.exists():
        print("Evaluating model on training data...")
        train_dataset = format_dataset(
            {
                "Y": res['Y'][:train_val_index],
                "U": res['U'][:train_val_index]
            },
            x_scaler, 
            u_scaler, 
            train=False,
            input_horizon=1,
            output_horizon=60,
            stride=5,
            name="train",
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        train_results = get_results(model, x_scaler, u_scaler, train_dataset)
        
        val_dataset = format_dataset(
            {
                "Y": res['Y'][train_val_index:val_test_index],
                "U": res['U'][train_val_index:val_test_index]
            },
            x_scaler,
            u_scaler,
            train=False,
            input_horizon=1,
            output_horizon=60,
            stride=5,
            name="val",
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        val_results = get_results(model, x_scaler, u_scaler, val_dataset)
        
        test_dataset = format_dataset(
            {
                "Y": res['Y'][val_test_index:],
                "U": res['U'][val_test_index:]
            },
            x_scaler,
            u_scaler,
            train=False,
            input_horizon=1,
            output_horizon=60,
            stride=5,
            name="test",
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        test_results = get_results(model, x_scaler, u_scaler, test_dataset)
        # combine all keys and values of the three results dicts
        train_results.update(val_results)
        train_results.update(test_results)
        np.savez(TRAINING_RESULTS_DATA, **train_results)
        
    # transfer data
    if not TRANSFER_RESULTS_PATH.exists():
        print("Evaluating Transfer No Maintenance Data...")
        res = dict(np.load(TRANSFER_DATA_PATH))
        for k, v in res.items():
            res[k] = v.astype(np.float64)
        results = {}
        
        for i,d in enumerate(MONTH_DAYS):
            month_start = sum(MONTH_DAYS[:i]) * 24 * 60
            month_end = month_start + d * 24 * 60
            dataset = format_dataset(
                {
                    "Y": res['Y'][month_start:month_end],
                    "U": res['U'][month_start:month_end]
                },
                x_scaler,
                u_scaler,
                train=False,
                input_horizon=1,
                output_horizon=60,
                stride=5,
                name=f"month_{i+1}",
                device=torch.get_default_device(),
                dtype=torch.get_default_dtype(),
            )
            month_results = get_results(model, x_scaler, u_scaler, dataset)
            results.update(month_results)
            for i in range(1):
                month_start = 0
                month_end = -1
                dataset = format_dataset(
                    {
                        "Y": res['Y'][month_start:month_end],
                        "U": res['U'][month_start:month_end]
                    },
                    x_scaler,
                    u_scaler,
                    train=False,
                    input_horizon=1,
                    output_horizon=60,
                    stride=5,
                    name=f"all",
                    device=torch.get_default_device(),
                    dtype=torch.get_default_dtype(),
                )
                month_results = get_results(model, x_scaler, u_scaler, dataset)
                results.update(month_results)
        np.savez(TRANSFER_RESULTS_PATH, **results)
        
        
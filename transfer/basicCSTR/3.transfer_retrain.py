# imports
from basicCSTR import *

# constants
method = "adam"

# script
if __name__ == "__main__":
    # ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFER_DATA_PATH = DATA_DIR.joinpath("transfer_data.npz")
    train_data = np.load(DATA_DIR.joinpath("training_data.npz"))
    transfer_data = np.load(TRANSFER_DATA_PATH)
    
    train_x_scale = train_data["Y"][:-61*24*60].std(axis=0)
    train_x_mean = train_data["Y"][:-61*24*60].mean(axis=0)
    train_u_scale = train_data["U"][:-61*24*60].std(axis=0)
    train_u_mean = train_data["U"][:-61*24*60].mean(axis=0)
    
    print(f"Train x scale: {train_x_scale}")
    print(f"Train x mean: {train_x_mean}")
    print(f"Train u scale: {train_u_scale}")
    print(f"Train u mean: {train_u_mean}")
    
    # ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFER_DATA_PATH = DATA_DIR.joinpath("transfer_data.npz")
    transfer_results = np.load(TRANSFER_DATA_PATH)
    
    for month in range(12):
        month_start = sum(MONTH_DAYS[:month]) * 24 * 60
        month_end = month_start + MONTH_DAYS[month] * 24 * 60
        for training_days in [1, 10]:
            for method in ["adam", "sekf", "lbfgs"]:
                TRANSFER_RESULTS_DIR = RESULTS_DIR.joinpath("transfer","retraining", f"month_{month+1}", "days_{training_days}", method)
                TRANSFER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                if not TRANSFER_RESULTS_DIR.joinpath("results.npz").exists():
                    train_validation_idx = month_start + 0.8 * (training_days * 24 * 60)
                    validation_test_idx = month_start + (training_days * 24 * 60)
                    data = {
                        
                    }
            
            
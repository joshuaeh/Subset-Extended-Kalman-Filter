
# imports
import pandas as pd

from dampedSpring import *

# constants

# script
if __name__ == "__main__":
    best_data = []
    
    transfer_dir = RESULTS_DIR.joinpath("transfer")
    for init_method in ["finetune", "retrain"]:
        initialization_dir = transfer_dir.joinpath(init_method)
        # initialization_dir.mkdir(parents=True, exist_ok=True)
        for scenario in SCENARIOS:
            scenario_name = transfer_scenario_name(scenario)
            scenario_dir = initialization_dir.joinpath(scenario_name)
            # scenario_dir.mkdir(parents=True, exist_ok=True)
            for data_dim in TARGET_DATA_DIM:
                data_dim_dir = scenario_dir.joinpath(data_dim_name(data_dim))
                # data_dim_dir.mkdir(parents=True, exist_ok=True)
                for data_iteration in range(N_ITERATIONS):
                    data_iteration_dir = data_dim_dir.joinpath(
                        get_data_iteration_name(data_iteration)
                    )
                    # data_iteration_dir.mkdir(parents=True, exist_ok=True)
                    
                    for update_method in ["adam", "sekf", "lbfgs"]:
                        update_method_dir = data_iteration_dir.joinpath(update_method)
                        results_file = update_method_dir.joinpath("bestResult_metrics.csv")
                        if results_file.exists():
                            df = pd.read_csv(results_file, index_col=0)
                            # df row with the lowest "val_loss" to dictionary
                            best_row = df.loc[df["val_loss"].idxmin()].to_dict()
                            best_row["init_method"] = init_method
                            best_row["scenario"] = scenario_name
                            best_row["data_dim"] = data_dim
                            best_row["data_iteration"] = data_iteration
                            
                        else:
                            best_row = {
                                "init_method": init_method,
                                "scenario": scenario_name,
                                "data_dim": data_dim,
                                "data_iteration": data_iteration,
                                "update_method": update_method,
                                "train_loss": np.nan,
                                "val_loss": np.nan,
                                "test_loss": np.nan,
                            }
                            
                        best_data.append(best_row)
                
    best_data_df = pd.DataFrame(best_data)
    best_data_df.to_csv(RESULTS_DIR.joinpath("transfer_bestResults.csv"))
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
import pandas as pd

from dampedSpring import *

MOVE_FILES = True
CLOBBER_EXISTING = False
MOVE_RAY_RESULTS = False

scratch_transfer_dir = Path("/scratch/08940/joshuaeh/Subset-Extended-Kalman-Filter/transfer/")
work_transfer_dir = Path("/work/08940/joshuaeh/projects/Updating Neural Networks/Subset-Extended-Kalman-Filter/transfer/")

results = []
best_data = []

n_complete = 0
n_total = 0

for init_method in ["retrain", "finetune"]:
    initialization_dir = RESULTS_DIR.joinpath("transfer", init_method)
    
    # --- check how many are complete ---
    n_complete = 0
    for data_iteration in range(N_ITERATIONS):
        for scenario in SCENARIOS:
            scenario_name = transfer_scenario_name(scenario)
            scenario_dir = initialization_dir.joinpath(scenario_name)
            # scenario_dir.mkdir(parents=True, exist_ok=True)
            for data_dim in TARGET_DATA_DIM:
                data_dim_dir = scenario_dir.joinpath(data_dim_name(data_dim))
                # data_dim_dir.mkdir(parents=True, exist_ok=True)

                data_iteration_dir = data_dim_dir.joinpath(
                    get_data_iteration_name(data_iteration)
                )
                # data_iteration_dir.mkdir(parents=True, exist_ok=True)
                
                for method in ["adam", "sekf", "lbfgs"]:
                    n_total += 1
                    
                    result = {
                        "init_method": init_method,
                        "scenario": scenario_name,
                        "data_dim": data_dim,
                        "data_iteration": data_iteration,
                        "method": method,
                    }
                    
                    best_row = {
                        "init_method": init_method,
                        "scenario": scenario_name,
                        "data_dim": data_dim,
                        "data_iteration": data_iteration,
                        "method": method,
                    }
                    
                    method_dir = data_iteration_dir.joinpath(method)
                    # method_dir.mkdir(parents=True, exist_ok=True)
                    
                    all_trials_path = method_dir.joinpath(ALL_TRIALS_BASE_FILENAME)
                    best_result_path = method_dir.joinpath(BEST_RESULT_BASE_FILENAME)
                    model_weights_path = method_dir.joinpath(MODEL_FILENAME)
                    ray_results_path = method_dir.joinpath("ray_results.zip")
                    if all(
                        [
                            all_trials_path.exists(),
                            best_result_path.exists(),
                            model_weights_path.exists(),
                            ray_results_path.exists(),
                        ]
                    ):
                        n_complete += 1
                        
                        result["complete"] = True
                        
                        if MOVE_FILES:
                            relative_dir_path = method_dir.relative_to(scratch_transfer_dir)
                            
                            
                            source_files = [
                                ALL_TRIALS_BASE_FILENAME,
                                BEST_RESULT_BASE_FILENAME,
                                MODEL_FILENAME
                            ]
                            if MOVE_RAY_RESULTS:
                                source_filepaths.append("ray_results.zip")
                                
                            for source_file in source_files:
                                source_filepath = method_dir.joinpath(source_file)
                                dest_filepath = work_transfer_dir.joinpath(relative_dir_path, source_file)
                                dest_filepath.parent.mkdir(parents=True, exist_ok=True)
                                if not dest_filepath.exists() or CLOBBER_EXISTING:
                                    logging.debug(f"Moving {source_filepath} to {dest_filepath}")
                                    shutil.copy2(
                                        source_filepath,
                                        dest_filepath,
                                    )
                            if all([
                                work_transfer_dir.joinpath(method_dir.joinpath(p).relative_to(scratch_transfer_dir)).exists() for p in source_files
                            ]):
                                logging.debug(f"Successfully moved files")
                                result["onWork"] = True
                            else:
                                logging.warning(f"Failed to move all files to work filesystem")
                                result["onWork"] = False
                        
                    else:
                        result["complete"] = False
                        
                    results.append(result)
                        
                    if best_result_path.exists():
                        best_result_df = pd.read_csv(best_result_path)
                        best_row.update(best_result_df.loc[best_result_df["val_loss"].idxmin()].to_dict())
                    else:
                        best_row.update({
                            "train_loss": None,
                            "val_loss": None,
                            "test_loss": None,
                        })
                    best_data.append(best_row)
                    
                    
for file in RESULTS_DIR.joinpath("training").glob("*"):
    if file.is_file():
        relative_path = file.relative_to(scratch_transfer_dir)
        dest_filepath = work_transfer_dir.joinpath(relative_path)
        dest_filepath.parent.mkdir(parents=True, exist_ok=True)
        if not dest_filepath.exists() or CLOBBER_EXISTING:
            logging.debug(f"Moving {file} to {dest_filepath}")
            shutil.copy2(
                file,
                dest_filepath,
            )
                    
logging.info(f"Completed {n_complete} of {n_total} transfer experiments.")

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR.joinpath("transfer_status.csv"), index=False)
results_df.to_csv(work_transfer_dir.joinpath(RESULTS_DIR.joinpath("transfer_status.csv").relative_to(scratch_transfer_dir)), index=False)

best_data_df = pd.DataFrame(best_data)
best_data_df.to_csv(RESULTS_DIR.joinpath("transfer_best_results.csv"), index=False)
best_data_df.to_csv(work_transfer_dir.joinpath(RESULTS_DIR.joinpath("transfer_best_results.csv").relative_to(scratch_transfer_dir)), index=False)
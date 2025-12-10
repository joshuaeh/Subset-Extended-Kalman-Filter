import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
import pandas as pd

from tclab_utils import *

MOVE_FILES = True
CLOBBER_EXISTING = False
MOVE_RAY_RESULTS = False

scratch_transfer_dir = Path("/scratch/08940/joshuaeh/Subset-Extended-Kalman-Filter/transfer/")
scratch_transfer_dir = Path("/home1/08940/joshuaeh/SCRATCH/Subset-Extended-Kalman-Filter/transfer/")
work_transfer_dir = Path("/work/08940/joshuaeh/projects/Updating Neural Networks/Subset-Extended-Kalman-Filter/transfer/")

results = []
best_data = []

n_complete = 0
n_total = 0

METHODS = {
    "SEKF": TCLabTrainer_SEKF, 
    "LBFGS": TCLabTrainer_LBFGS, 
    "Adam": TCLabTrainer,
}

for day in range(DAYS): 
    for data_dim_name, data_dim in DATA_DIM.items():
        for initialization_name, initialization_config in INITIALIZATIONS.items():
            for method, trainer in METHODS.items():
                    
                TRANSFER_RESULTS_DIR = RESULTS_DIR.joinpath(initialization_name, f"day{day}", data_dim_name, method)
                TRANSFER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                
                all_trials_path = TRANSFER_RESULTS_DIR.joinpath(ALL_TRIALS_BASE_FILENAME)
                best_result_path = TRANSFER_RESULTS_DIR.joinpath(BEST_RESULT_BASE_FILENAME)
                model_weights_path = TRANSFER_RESULTS_DIR.joinpath(MODEL_FILENAME)
                ray_results_path = TRANSFER_RESULTS_DIR.joinpath("ray_results.zip")
                
                result = {
                    "init_method": initialization_name,
                    "day": day+1,
                    "data_dim": data_dim_name,
                    "method": method,
                    "complete": all([
                        all_trials_path.exists(),
                        best_result_path.exists(),
                        model_weights_path.exists(),
                        ray_results_path.exists()
                    ])}
                
                best_row = {
                    "init_method": initialization_name,
                    "day": day+1,
                    "data_dim": data_dim_name,
                    "method": method,
                }
                
                if result["complete"]:
                    n_complete += 1
                
                    if MOVE_FILES:
                        relative_dir_path = TRANSFER_RESULTS_DIR.relative_to(scratch_transfer_dir)
                        
                        source_files = [
                            ALL_TRIALS_BASE_FILENAME,
                            BEST_RESULT_BASE_FILENAME,
                            MODEL_FILENAME,
                        ]
                        if MOVE_RAY_RESULTS:
                            source_files.append("ray_results.zip")
                        for source_file in source_files:
                            source_filepath = scratch_transfer_dir.joinpath(relative_dir_path, source_file)
                            dest_filepath = work_transfer_dir.joinpath(relative_dir_path, source_file)
                            dest_filepath.parent.mkdir(parents=True, exist_ok=True)
                            if not dest_filepath.exists() or CLOBBER_EXISTING:
                                logging.debug(f"Moving {source_filepath} to {dest_filepath}")
                                shutil.copy2(
                                    source_filepath,
                                    dest_filepath,
                                )
                        if all([
                            work_transfer_dir.joinpath(relative_dir_path, p).exists() for p in source_files
                        ]):
                            logging.debug(f"Successfully moved files")
                            result["onWork"] = True
                        else:
                            result["onWork"] = False
                n_total += 1
                results.append(result)
                
                if best_result_path.exists():
                    best_result_df = pd.read_csv(best_result_path)
                    best_row.update(best_result_df.loc[best_result_df["val_L2e"].idxmin()].to_dict())
                else:
                    best_row.update({
                        "train_L2e": None,
                        "val_L2e": None,
                        "test_L2e": None,
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
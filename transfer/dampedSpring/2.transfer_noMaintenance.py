"""Evaluate trained model on transfer data without maintenance (no further training)."""

# imports
from dampedSpring import *

# constants

# script
if __name__ == "__main__":
    # declare case_dir and create if doesn't exist
    case_dir = RESULTS_DIR.joinpath("transfer", "noMaintenance")
    if not case_dir.exists():
        case_dir.mkdir(parents=True, exist_ok=True)

    NN = MLP()
    NN.load_state_dict(torch.load(MODEL0_WEIGHTS_PATH))
    NN.eval()

    results = {}

    for scenario in SCENARIOS:
        scenario_name = transfer_scenario_name(scenario)
        print(f"Evaluating scenario: {scenario_name}")

        x, y = get_transfer_data(scenario)
        yp = NN(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        results[scenario_name + "-y"] = y
        results[scenario_name + "-yp"] = yp
    results["x"] = x

    # save results
    np.savez(case_dir.joinpath("results.npz"), results)

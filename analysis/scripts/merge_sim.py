import json
import glob
from pathlib import Path

def merge_simulation_files(model_type):
    """Merge individual simulation files into a single file per condition"""
    base_path = Path("../data")
    for variant in ["R", "V", "T"]:
        folder = str(base_path / f"simulated_{model_type}")
        all_data = {}
        for file in glob.glob(folder + f"/{variant}/*_data.json"):
            with open(file) as f:
                user = file.split("user")[1].split("_")[0]
                data = json.load(f)
                all_data[f"user{user}"] = data

        all_params = {}
        for file in glob.glob(folder + f"/{variant}/*_params.json"):
            with open(file) as f:
                user = file.split("user")[1].split("_")[0]
                params = json.load(f)
                all_params[f"user{user}"] = params

        with open(f"{folder}/data_{variant}.json", "w") as f:
            json.dump(all_data, f)

        with open(f"{folder}/params_{variant}.json", "w") as f:
            json.dump(all_params, f)

if __name__ == "__main__":
    # Merge files for both model types
    merge_simulation_files("filter_adapt")
    merge_simulation_files("policy_compress")
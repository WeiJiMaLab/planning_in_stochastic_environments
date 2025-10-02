import json
import glob
for folder in glob.glob("../data/simulated_filteradapt/*"):
    all_data = {}
    for file in glob.glob(folder + "/*.json"):
        with open(file, "r") as f:
            user = file.split("user")[1].split("_")[0]
            data = json.load(f)
            all_data["user" + user] = data
            print(user)
    
    with open(f"{folder}.json", "w") as f:
        print(folder.split('/')[-1])
        json.dump(all_data, f)

for folder in glob.glob("../data/simulated_policycomp/*"):
    all_data = {}
    for file in glob.glob(folder + "/*.json"):
        with open(file, "r") as f:
            user = file.split("user")[1].split("_")[0]
            data = json.load(f)
            all_data["user_" + user] = data
            print(user)
    
    with open(f"{folder}.json", "w") as f:
        print(folder.split('/')[-1])
        json.dump(all_data, f)
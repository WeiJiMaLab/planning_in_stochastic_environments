import os
import sys
import inspect
import tqdm
import warnings
import argparse
import datetime

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 
from modeling import *
from utils import NpEncoder


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--type", default="R")
args = parser.parse_args()
type_ = args.type

# get job id
jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
print("Job ID\t", jobid)


# load data and get player
datafile = "../data/raw/data_%s.json"%(type_)
print("data file\t", datafile)
with open(datafile, 'r') as f:
    data = json.load(f)
player = list(data.keys())[jobid]


# multi-start parameters
params = {
"inv_temp": 5, 
}

# params.update({p:5 for p in get_conditions(type_)})

filter_fns = [filter_depth, filter_rank, filter_value]
value_fns = [value_EV, value_path, value_max, value_sum]


# fit models with multi-start
with warnings.catch_warnings():
    # filter warnings
    warnings.simplefilter("ignore")
    
    games = format_games(data[player]["data"])
    
    for filter_fn in filter_fns:
        for value_fn in value_fns:
            if type_ == "V" and value_fn == value_EV: 
                pass
            else:
                model = Model(filter_fn, value_fn, variant = type_)
                filedir = f"../data/fit_{datetime.datetime.now().strftime('%Y%m%d')}/{type_}_{model.name}"
                filename = f"{filedir}/{player}.json"
                print("saving to:\t", filename)
                
                multistart = MultiStart(model, games, params, use_grid = False, n=100)
                multistart.sweep()
                
                try: os.makedirs(filedir)
                except FileExistsError: pass
                
                with open(filename, "w") as f:
                    json.dump(multistart.best, f, cls=NpEncoder)

                # save parameters
                with open(f"{filedir}/params.json", "w") as f: 
                    json.dump(params, f)

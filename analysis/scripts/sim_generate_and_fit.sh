#!/bin/bash

# Example of running python script with a job array
#SBATCH -J sim_gen_fit
#SBATCH -c 1 # one CPU core per task
#SBATCH -t 4:00:00
#SBATCH --mem=4GB
#SBATCH --array=0-20
#SBATCH -o log/%j_%a.out
#SBATCH -e log/%j_%a.error 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl3976@nyu.edu  
#SBATCH --job-name=TH-sim-gen-fit

# activate the venv
source /scratch/hl3976/ma_lab/.venv/bin/activate

# run the python script
python sim_generate_and_fit.py --type=$1

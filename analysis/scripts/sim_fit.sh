#!/bin/bash

# Example of running python script with a job array
#SBATCH -J model_fit
#SBATCH -c 1 # one CPU core per task
#SBATCH -t 2:00:00
#SBATCH --mem=2GB
#SBATCH --array=0-20
#SBATCH -o log/%j_%a.out
#SBATCH -e log/%j_%a.error 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl3976@nyu.edu  
#SBATCH --job-name=TH-fit

# activate the venv
source /scratch/hl3976/ma_lab/.venv/bin/activate

# run the python script
python sim_fit.py --type=$1
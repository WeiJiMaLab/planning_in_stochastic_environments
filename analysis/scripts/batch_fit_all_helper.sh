#!/bin/bash

# Example of running python script with a job array
#SBATCH -J model_fit_${1}_${2}
#SBATCH -c 1 # one CPU core per task
#SBATCH -t 2-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-99
#SBATCH -o log/%j_${1}_${2}_%a.out
#SBATCH -e log/%j_${1}_${2}_%a.error 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl3976@nyu.edu  
#SBATCH --job-name=Fit_${1}_${2}

# activate the venv
source /scratch/hl3976/ma_lab/.venv/bin/activate

# run the python script
python batch_fit_all.py --type=$1 --data_folder=$2
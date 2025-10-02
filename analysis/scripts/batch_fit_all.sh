#!/bin/bash
rm -rf log/*
sbatch batch_fit_all_array.sh R raw
sbatch batch_fit_all_array.sh T raw
sbatch batch_fit_all_array.sh V raw
sbatch batch_fit_all_array.sh R sim_variable_depth
sbatch batch_fit_all_array.sh T sim_variable_depth
sbatch batch_fit_all_array.sh V sim_variable_depth
sbatch batch_fit_all_array.sh R sim_variable_inv_temp
sbatch batch_fit_all_array.sh T sim_variable_inv_temp
sbatch batch_fit_all_array.sh V sim_variable_inv_temp

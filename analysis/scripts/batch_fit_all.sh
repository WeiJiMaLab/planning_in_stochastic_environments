#!/bin/bash
rm -rf log/*
sbatch batch_fit_all_array.sh R raw
sbatch batch_fit_all_array.sh T raw
sbatch batch_fit_all_array.sh V raw
sbatch batch_fit_all_array.sh R simulated_filteradapt
sbatch batch_fit_all_array.sh T simulated_filteradapt
sbatch batch_fit_all_array.sh V simulated_filteradapt
sbatch batch_fit_all_array.sh R simulated_policycomp
sbatch batch_fit_all_array.sh T simulated_policycomp
sbatch batch_fit_all_array.sh V simulated_policycomp

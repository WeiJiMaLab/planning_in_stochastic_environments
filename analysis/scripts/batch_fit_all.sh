#!/bin/bash
rm -rf log/*
sbatch batch_fit_all_helper.sh R raw
sbatch batch_fit_all_helper.sh T raw
sbatch batch_fit_all_helper.sh V raw
sbatch batch_fit_all_helper.sh R simulated_filter_adapt
sbatch batch_fit_all_helper.sh T simulated_filter_adapt
sbatch batch_fit_all_helper.sh V simulated_filter_adapt
sbatch batch_fit_all_helper.sh R simulated_policy_compress
sbatch batch_fit_all_helper.sh T simulated_policy_compress
sbatch batch_fit_all_helper.sh V simulated_policy_compress

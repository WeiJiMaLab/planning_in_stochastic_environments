#!/bin/bash
rm -rf log/*
sbatch sim_fit.sh R
sbatch sim_fit.sh T
sbatch sim_fit.sh V
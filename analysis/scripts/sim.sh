#!/bin/bash
rm -rf log/*
rm -rf ../data/sim/
sbatch sim_fit.sh R
sbatch sim_fit.sh T
sbatch sim_fit.sh V
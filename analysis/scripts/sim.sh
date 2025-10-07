#!/bin/bash
rm -rf log/*
sbatch sim_helper.sh R
sbatch sim_helper.sh T
sbatch sim_helper.sh V
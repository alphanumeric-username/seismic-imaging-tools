#!/bin/bash

module load anaconda3/2022.10
module load openmpi/4.0.7-gcc-ucx-1.13.1

source ~/.bashrc

conda deactivate
conda activate seismage
cd $CWD

mpirun python rtm.py $@
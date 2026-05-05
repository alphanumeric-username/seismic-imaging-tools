#!/bin/bash

module load anaconda3/2022.10
module load openmpi/4.0.7-gcc-ucx-1.13.1

# source activate devito
source ~/.bashrc
#conda init bash

conda deactivate
conda activate pylops
cd $CWD

# export DEVITO_LANGUAGE=openmp

# mpirun -np 5 
mpirun python fwd.py $@
#!/bin/bash

module load anaconda3/2022.10
module load openmpi/4.0.7-gcc-ucx-1.13.1

#conda init bash
source ~/.bashrc

# source activate devito
conda deactivate
conda activate pylops
# conda activate pylops_mpi_clone
cd $CWD

# export DEVITO_LANGUAGE=openmp

# mpirun -np 5 
mpirun python fwd.py $@
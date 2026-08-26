#!/bin/sh


# Modeling parameters
MODELFILE=examples/marmousi/true_model/true.yml
GEOMETRYFILE=examples/marmousi/geometry.yml  
PDEFILE=solvers/acoustic_solver.py
NAME=dobs
OUTDIR=out/fwd


# SLURM parameters
NTASKS=50
NTASKS_PER_NODE=25

ACCOUNT=geo-inct
PARTITION=standard


# Program call
mkdir $OUTDIR -p
CWD=$(pwd) sbatch --job-name="fwd:"$NAME --ntasks=$NTASKS --ntasks-per-node=$NTASKS_PER_NODE -A $ACCOUNT -p $PARTITION -o $OUTDIR/$NAME.log model-data-node.sh $MODELFILE -o $OUTDIR -n $NAME -g $GEOMETRYFILE -e $PDEFILE
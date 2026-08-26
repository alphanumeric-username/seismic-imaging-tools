#!/bin/sh


# Modeling parameters
MODELFILE=examples/marmousi/smooth_model/smooth.yml  # Smooth model
GEOMETRYFILE=examples/marmousi/geometry.yml
PDEFILE=solvers/acoustic_solver.py
DOBS=out/fwd/dobs.bin
NAME=rtm
OUTDIR=out/rtm/


# SLURM parameters
NTASKS=30
NTASKS_PER_NODE=10

ACCOUNT=geo-inct
PARTITION=standard


mkdir $OUTDIR -p
CWD=$(pwd) sbatch --job-name="rtm:"$NAME --ntasks=$NTASKS --ntasks-per-node=$NTASKS_PER_NODE -A $ACCOUNT -p $PARTITION -o $OUTDIR/$NAME.log rtm-node.sh $MODELFILE -o $OUTDIR/$NAME.bin -g $GEOMETRYFILE -e $PDEFILE -d $DOBS
#!/bin/sh

MODELFILE=examples/marmousi/model/true.yml
GEOMETRYFILE=examples/marmousi/geometry.yml
PDEFILE=solvers/acoustic_solver.py
NAME=dobs_test
OUTDIR=out/fwd

NTASKS=100
NTASKS_PER_NODE=25

ACCOUNT=geo-inct
PARTITION=standard

CWD=$(pwd) sbatch --job-name="fwd:"$NAME --ntasks=$NTASKS --ntasks-per-node=$NTASKS_PER_NODE -A $ACCOUNT -p $PARTITION -o $OUTDIR/$NAME.log fwd-node.sh $MODELFILE -o $OUTDIR -n $NAME -g $GEOMETRYFILE -e $PDEFILE
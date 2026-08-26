import sys, argparse, os
from typing import List, Tuple
import yaml
import numpy as np

from mpi4py import MPI
import seismagelib.data_io as sio
from seismagelib.wavesolver.acoustic import create_solver_class
from seismagelib.waveeq_processing.generic_acoustic_twoway import GenericAcousticWave2D

import time, datetime

RANK_MASTER = 0

def main(argv):
    wcomm = MPI.COMM_WORLD
    args = parse_args(argv)
    modelparams = sio.load_model(args.modelfile)
    geoparams = sio.load_geometry(args.geometryfile)

    if wcomm.rank == RANK_MASTER:
        t0 = time.time()
    
    nx = modelparams['shape'][0]
    dx = modelparams['spacing'][0]
    nsrc = geoparams['ns']
    nrec = geoparams['nr']

    dt = geoparams['dt']
    tn = geoparams['tn']
    fpeak = geoparams.get('fpeak', 10)

    max_jobs, shot_intervals = gen_shot_intervals(nsrc, wcomm.size)
    
    if wcomm.size > max_jobs:
        print(max_jobs, wcomm.size)
        if wcomm.rank == RANK_MASTER:
            log(f'Too much processes!\n\tThere are {wcomm.size} processes but only {nsrc} shots.')
        return 1

    isrc_min, isrc_max = shot_intervals[wcomm.rank]

    dobs_local = np.fromfile(args.dobs, dtype=np.float32)
    dobs_local = dobs_local.reshape((nsrc, nrec, dobs_local.shape[0]//(nsrc*nrec)))[isrc_min:isrc_max]

    print(f'{log_handle()}: I will compute shots {isrc_min} to {isrc_max}')

    pdemodule = sio.import_module_file(args.pdefile)
    parameter_names = list(pdemodule.PARAMETERS)

    solver_cls = create_solver_class(pdemodule.forward, pdemodule.adjoint, parameter_names, pdemodule.gradient)

    def_src_x = np.linspace(0, (nx - 1)*dx, num=nx, dtype=np.float32)
    def_src_z = np.zeros(nx)
    def_rec_x = np.linspace(0, (nx - 1)*dx, num=nx, dtype=np.float32)
    def_rec_z = np.zeros(nx)
    
    op_args = {
        'shape': modelparams['shape'],
        'origin': modelparams.get('origin', (0, 0)),
        'spacing': modelparams['spacing'],
        'solver_cls': solver_cls,
        'space_order': 8,
        'nbl': modelparams.get('nbl', 40),
        'src_x': geoparams.get('src', {'x': def_src_x})['x'][isrc_min:isrc_max],
        'src_z': geoparams.get('src', {'z': def_src_z})['z'][isrc_min:isrc_max],
        'rec_x': geoparams.get('rec', {'x': def_rec_x})['x'],
        'rec_z': geoparams.get('rec', {'z': def_rec_z})['z'],
        'src_type': geoparams['wavelet'] if type(geoparams['wavelet']) == str else 'Ricker',
        't0': 0,
        'tn': tn,
        'dt': dt,
        'dtype': np.float32,
        'f0': fpeak,
        'op_name': 'fwd',
        'parameter_names': parameter_names,
        'params': np.array([
            modelparams['params'][pname] for pname in parameter_names
        ], dtype=np.float32)
    }

    Aop = GenericAcousticWave2D(**op_args)
    
    if type(geoparams['wavelet']) == np.ndarray:
        Aop.updatesrc(geoparams['wavelet'])
    
    
    grad_local = Aop.H * dobs_local
    
    print(f'{log_handle()}: Done')


    grad_stack = np.zeros_like(grad_local).reshape(-1)

    wcomm.Reduce(
        [grad_local.reshape(-1), MPI.FLOAT],
        [grad_stack, MPI.FLOAT],
        MPI.SUM,
        RANK_MASTER
    )

    if wcomm.rank == RANK_MASTER:
        print(f'{log_handle()}: Writing to disk...')
        grad_stack = grad_stack.reshape(grad_local.shape)

        grad_stack[0].tofile(args.outfile)
        print(f'{log_handle()}: Done')
    
        dt = time.time() - t0
        print(f'{log_handle()}: Elapsed time: {datetime.timedelta(seconds=dt)}.')

    return 0


def log_handle():
    if MPI.COMM_WORLD.rank == RANK_MASTER:
        return '[MASTER]'
    else:
        return f'[WORKER_{MPI.COMM_WORLD.rank}]'


def log(str):
    print(f'{log_handle()}: {str}')


def gen_shot_intervals(nshots: int, nworkers: int) -> List[Tuple[int, int]]:
    max_jobs = min(nworkers, nshots)

    intervals = []
    indices = np.linspace(0, nshots, max_jobs+1, dtype=np.int32)
    for i in range(1, max_jobs + 1):
        intervals.append((int(indices[i-1]), int(indices[i])))

    return max_jobs, intervals


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('modelfile',            type=str               )
    parser.add_argument('--outfile',       '-o', type=str, required=True)
    parser.add_argument('--geometryfile', '-g', type=str, required=True)
    parser.add_argument('--pdefile',      '-e', type=str, required=True)
    parser.add_argument('--dobs',         '-d', type=str, required=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    # try:
    # except Exception as err:
    #     print(err.with_traceback(None))
    #     MPI.COMM_WORLD.Abort()

import sys, argparse, os
from typing import List, Tuple
import yaml
import numpy as np

from mpi4py import MPI
import seismagelib.data_io as sio
from seismagelib.wavesolver.acoustic import create_solver_class
from seismagelib.waveeq_processing.generic_acoustic_twoway import GenericAcousticWave2D

import time, datetime

# from pylops_mpi.DistributedArray import local_split, Partition

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

    dt = modelparams['dt']
    tn = geoparams['tn']
    f0 = geoparams['f0']

    # ns_local = local_split((nsrc,), wcomm, Partition.SCATTER, 0)
    # ns_alllocals = np.concatenate(wcomm.allgather(ns_local))
    # isrc_min = np.concatenate([[0], np.cumsum(ns_alllocals)[:-1]])[wcomm.rank]
    # isrc_max = np.cumsum(ns_alllocals)[wcomm.rank]

    max_jobs, shot_intervals = gen_shot_intervals(nsrc, wcomm.size)
    
    if wcomm.size > max_jobs:
        if wcomm.rank == RANK_MASTER:
            log(f'Too much processes!\n\tThere are {wcomm.size} processes but only {nsrc} shots.')
        return 1

    isrc_min, isrc_max = shot_intervals[wcomm.rank]

    print(f'{log_handle()}: I will compute shots {isrc_min} to {isrc_max}')
    # parameter_names = [ s.strip() for s in args.parameters.split(',') ]

    pdemodule = sio.import_module_file(args.pdefile)
    parameter_names = list(pdemodule.PARAMETERS)


    solver_cls = create_solver_class(pdemodule.forward, pdemodule.adjoint, parameter_names)



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
        'src_type': geoparams['source'] if type(geoparams['source']) == str else 'Ricker',
        't0': 0,
        'tn': tn,
        'dt': dt,
        'dtype': np.float32,
        'f0': f0,
        'op_name': 'fwd',
        'parameter_names': parameter_names,
        'params': np.array([
            modelparams['params'][pname] for pname in parameter_names
        ], dtype=np.float32)
    }

    # Aop = resolve_operator_class(args.solver)(**op_args)
    Aop = GenericAcousticWave2D(**op_args)
    if type(geoparams['source']) == np.ndarray:
        Aop.updatesrc(geoparams['source'])
    
    dobs_local = Aop @ op_args['params']
    
    
    print(f'{log_handle()}: Done')

    alldobs = gather_shots(wcomm, nsrc, isrc_min, isrc_max, dobs_local)

    if wcomm.rank == RANK_MASTER:
        print(f'{log_handle()}: Writing to disk...')
        dobs = alldobs.reshape((nsrc, *dobs_local[0].shape))

        dobs.tofile(os.path.join(args.outdir, args.name + '.bin'))
        geopath = os.path.relpath(args.geometryfile, args.outdir)
        with open(os.path.join(args.outdir, args.name + '.yml'), 'w+') as fout:
            yaml.dump({
                'ns': nsrc,
                'nt': dobs_local.shape[2],
                'tn': tn,
                'dt': dt,
                'f0': f0,
                'geometry': geopath,
                'data': './' + args.name + '.bin'
            }, fout)
        
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

    shots_per_job = int(np.ceil(nshots / max_jobs))
    return max_jobs, [
        (
            i*shots_per_job, 
            min((i+1)*shots_per_job, nshots)
        ) 
        for i in range(nworkers)
    ]

def gather_shots(wcomm: MPI.Intracomm, nsrc, isrc_min, isrc_max, dobs_local):
    n_processes_with_remaining_shots = np.array([0], dtype=np.int32)
    alldobs = None
    if wcomm.rank == RANK_MASTER:
        alldobs = np.zeros((nsrc, *dobs_local[0].shape), dtype=np.float32)

    curr_isrc = isrc_min
    are_there_remaining_shots = np.array([curr_isrc < isrc_max], dtype=np.int32)

    wcomm.Allreduce(
        [are_there_remaining_shots, MPI.INT],
        [n_processes_with_remaining_shots, MPI.INT]
    )

    while n_processes_with_remaining_shots[0] > 0:
        partial_dobs = np.zeros(wcomm.size * np.prod(dobs_local[0].shape), dtype=np.float32)
        partial_indexes = np.zeros(wcomm.size, dtype=np.int32)
        
        if are_there_remaining_shots[0]:
            wcomm.Gather(
                [dobs_local[curr_isrc - isrc_min].reshape(-1), MPI.FLOAT],
                [partial_dobs, MPI.FLOAT], RANK_MASTER
            )
        else:
            wcomm.Gather(
                [dobs_local[0].reshape(-1), MPI.FLOAT],
                [partial_dobs, MPI.FLOAT], RANK_MASTER
            )
        
        partial_dobs = partial_dobs.reshape((wcomm.size, *dobs_local[0].shape))
        
        if are_there_remaining_shots[0]:
            idx = np.array([curr_isrc], dtype=np.int32)
        else:
            idx = np.array([-1], dtype=np.int32)

        wcomm.Gather(
            [idx, MPI.INT],
            [partial_indexes, MPI.INT], RANK_MASTER
        )

        if wcomm.rank == RANK_MASTER:
            for j in range(partial_indexes.shape[0]):
                i = partial_indexes[j]
                if i >= 0:
                    alldobs[i] = partial_dobs[j]

        curr_isrc += 1
        are_there_remaining_shots = np.array([curr_isrc < isrc_max], dtype=np.int32)
        wcomm.Allreduce(
            [are_there_remaining_shots, MPI.INT],
            [n_processes_with_remaining_shots, MPI.INT], MPI.SUM
        )

    return alldobs


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('modelfile',            type=str               )
    parser.add_argument('--outdir',       '-o', type=str, required=True)
    parser.add_argument('--name',         '-n', type=str, required=True)
    parser.add_argument('--geometryfile', '-g', type=str, required=True)
    parser.add_argument('--pdefile',      '-e', type=str, required=True)
    # parser.add_argument('--parameters',   '-p', type=str, required=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as err:
        print(err.with_traceback())
        MPI.COMM_WORLD.Abort()

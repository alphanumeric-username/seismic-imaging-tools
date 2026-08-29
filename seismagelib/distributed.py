from typing import List, Tuple

from mpi4py import MPI

import numpy as np


RANK_MASTER = 0


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
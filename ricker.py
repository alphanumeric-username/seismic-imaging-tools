import argparse
import sys

import numpy as np

def main(argv):
    args = parse_args(argv)

    s = ricker(args.fpeak, args.dt, int(args.tn//args.dt))

    s.tofile(args.outfile)

    return 0


def ricker(fpeak, dt, nt):
    t = np.arange(nt) * dt
    t0 = 1/fpeak
    t_ = t - t0
    return np.array((1 - 2 * np.pi**2 * fpeak**2 * t_**2) * np.exp(-np.pi**2 * fpeak**2 * t_**2), dtype=np.float32)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpeak', '-f', type=float, required=True)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--tn', type=float, required=True)
    parser.add_argument('--outfile', '-o', type=str, required=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from crank.DihedralScanner import DihedralScanner, get_geo_key
from crank.QMEngine import QMEngine, EnginePsi4, EngineQChem, EngineTerachem

def generate_task_cache(finished_jobs):
    """
    Take a dictionary of finished optimizations, generate the task_cache dictionary

    Parameters:
    ------------
    finished_jobs = dict(), key is the grid_id, value is a tuple of (job_id, start_geo, end_geo, end_energy)

    Returns:
    ------------
    task_cache = dict(), key is the grid_id, value is a dictionary {geo_key: (end_geo, end_energy, job_id)}
    Note: job_id here is used to determine which optimization ran first.
    """


def get_next_jobs(current_state):
    """
    Take current scan state and generate the next set of optimizations.
    This function will create a new DihedralScanner object and read all information from current_state,
    then reproduce the entire scan from the beginning, finish all cached ones, until a new job is not found in the cache.
    Return a list of new jobs that needs to be finished for the current iteration

    Input:
    -------
    current_state: dict, e.g. {
            'dihedrals': [[0,1,2,3], [1,2,3,4]] ,
            'grid_spacing': 30,
            'elems': ['H', 'C', 'O', ...]
            'init_coords': [geo1, geo2, ..]
            'finished_jobs': dict
        }

    Output:
    -------
    next_jobs: list, e.g. [

    ]
    """
    scanner = DihedralScanner(QMEngine(), dihedrals, grid_spacing, init_coords_M, verbose=True)
    scanner.task_cache = current_state['task_cache']


def main():
    import argparse, sys
    parser = argparse.ArgumentParser(description="Potential energy scan of dihedral angle from 1 to 360 degree", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile', type=str, help='Input template file for QMEngine. Geometry will be used as starting point for scanning.')
    parser.add_argument('dihedralfile', type=str, help='File defining all dihedral angles to be scanned.')
    parser.add_argument('--init_coords', type=str, help='File contain a list of geometries, that will be used as multiple starting points, overwriting the geometry in input file.')
    parser.add_argument('-g', '--grid_spacing', type=int, default=15, help='Grid spacing for dihedral scan, i.e. every 15 degrees')
    parser.add_argument('-e', '--engine', type=str, default="psi4", choices=['qchem', 'psi4', 'terachem'], help='Engine for running scan')
    parser.add_argument('--native_opt', action='store_true', default=False, help='Use QM program native constrained optimization algorithm. This will turn off geomeTRIC package.')
    parser.add_argument('--wq_port', type=int, default=None, help='Specify port number to use Work Queue to distribute optimization jobs.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print more information while running.')
    args = parser.parse_args()

    # print input command for reproducibility
    print(' '.join(sys.argv))

    # parse the dihedral file
    dihedral_idxs = load_dihedralfile(args.dihedralfile)

    # create QM Engine, and WorkQueue object if provided port
    engine = create_engine(args.engine, inputfile=args.inputfile, work_queue_port=args.wq_port, native_opt=args.native_opt)

    from geometric.molecule import Molecule
    # load init_coords if provided
    init_coords_M = Molecule(args.init_coords) if args.init_coords else None

    # create DihedralScanner object
    engine, dihedrals=dihedral_idxs, grid_spacing=args.grid_spacing, init_coords_M=init_coords_M, verbose=args.verbose)
    # Run the scan!
    scanner.master()
    # After finish, print result
    print("Dihedral scan is finished!")
    print(" Grid ID                Energy")
    for grid_id in sorted(scanner.grid_energies.keys()):
        print("  %-20s %.10f" % (str(grid_id), scanner.grid_energies[grid_id]))

def test():
    engine = create_engine('psi4')
    for dim in range(1, 4):
        print("Testing %d-D scan setup" % dim)
        dihedrals = [list(range(d, d+4)) for d in range(dim)]
        scanner = DihedralScanner(engine, dihedrals=dihedrals, grid_spacing=90)
        gid = scanner.grid_ids[0]
        assert len(scanner.grid_ids) == 4**dim and len(gid) == dim
        assert len(scanner.grid_neighbors(gid)) == 2**dim
    print("All tests passed!")


if __name__ == "__main__":
    main()
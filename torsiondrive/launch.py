#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from torsiondrive.dihedral_scanner import DihedralScanner
from torsiondrive.qm_engine import EnginePsi4, EngineQChem, EngineTerachem
from geometric.molecule import Molecule


def load_dihedralfile(dihedralfile, zero_based_numbering=False):
    """
    Load definition of dihedral from a text file, i.e. Loading the file

    # dihedral definition by atom indices starting from 0
    # i     j     k     j
      1     2     3     4
      2     3     4     5

    Will return dihedral_idxs = [(0,1,2,3), (1,2,3,4)]

    If a comment line
    #zero_based_numbering
    is found at the beginning of the file, or
    parameter zero_based_numbering == True
    reading will become

    # i     j     k     j
      1     2     3     4
      2     3     4     5

    Returns dihedral_idxs = [(1,2,3,4), (2,3,4,5)]

    """
    dihedral_idxs = []
    with open(dihedralfile) as infile:
        for line in infile:
            line = line.strip()
            if not line: continue
            if line[0] == '#':
                comment = line[1:].strip().lower()
                if comment == 'zero_based_numbering':
                    zero_based_numbering = True
                elif comment == 'one_based_numbering':
                    if zero_based_numbering == True:
                        raise ValueError("Can not specify both zero_based_numbering and one_based_indexing.")
                continue
            if zero_based_numbering == False:
                dihedral_idxs.append([int(i)-1 for i in line.split()])
            else:
                dihedral_idxs.append([int(i) for i in line.split()])
    return dihedral_idxs

def create_engine(enginename, inputfile=None, work_queue_port=None, native_opt=False):
    """
    Function to create a QM Engine object with work_queue and geomeTRIC setup.
    This is intentionally left outside of DihedralScanner class, because multiple DihedralScanner could share the same engine
    """
    engine_dict = {'psi4': EnginePsi4, 'qchem': EngineQChem, 'terachem':EngineTerachem}
    # initialize a work_queue
    if work_queue_port is not None:
        from torsiondrive.wq_tools import WorkQueue
        work_queue = WorkQueue(work_queue_port)
    else:
        work_queue = None
    engine = engine_dict[enginename](inputfile, work_queue, native_opt=native_opt)
    return engine

def main():
    import argparse, sys
    parser = argparse.ArgumentParser(description="Potential energy scan of dihedral angle from 1 to 360 degree", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile', type=str, help='Input template file for QMEngine. Geometry will be used as starting point for scanning.')
    parser.add_argument('dihedralfile', type=str, help='File defining all dihedral angles to be scanned.')
    parser.add_argument('--init_coords', type=str, help='File contain a list of geometries, that will be used as multiple starting points, overwriting the geometry in input file.')
    parser.add_argument('-g', '--grid_spacing', type=int, nargs='*', default=[15], help='Grid spacing for dihedral scan, i.e. every 15 degrees, multiple values will be mapped to each dihedral angle')
    parser.add_argument('-e', '--engine', type=str, default="psi4", choices=['qchem', 'psi4', 'terachem'], help='Engine for running scan')
    parser.add_argument('--native_opt', action='store_true', default=False, help='Use QM program native constrained optimization algorithm. This will turn off geomeTRIC package.')
    parser.add_argument('--wq_port', type=int, default=None, help='Specify port number to use Work Queue to distribute optimization jobs.')
    parser.add_argument('--zero_based_numbering', action='store_true', help='Use zero_based_numbering in dihedrals file.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print more information while running.')
    args = parser.parse_args()

    # print input command for reproducibility
    print(' '.join(sys.argv))

    # parse the dihedral file
    dihedral_idxs = load_dihedralfile(args.dihedralfile, args.zero_based_numbering)
    grid_dim = len(dihedral_idxs)

    # format grid spacing
    n_grid_spacing = len(args.grid_spacing)
    if n_grid_spacing == grid_dim:
        grid_spacing = args.grid_spacing
    elif n_grid_spacing == 1:
        grid_spacing = args.grid_spacing * grid_dim
    else:
        raise ValueError("Number of grid_spacing values %d is not consistent with number of dihedral angles %d" % (grid_dim, n_grid_spacing))

    # create QM Engine, and WorkQueue object if provided port
    engine = create_engine(args.engine, inputfile=args.inputfile, work_queue_port=args.wq_port, native_opt=args.native_opt)

    # load init_coords if provided
    init_coords_M = Molecule(args.init_coords) if args.init_coords else None

    # create DihedralScanner object
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=grid_spacing, init_coords_M=init_coords_M, verbose=args.verbose)
    # Run the scan!
    scanner.master()
    # After finish, print result
    print("Dihedral scan is finished!")
    print(" Grid ID                Energy")
    for grid_id in sorted(scanner.grid_energies.keys()):
        print("  %-20s %.10f" % (str(grid_id), scanner.grid_energies[grid_id]))

if __name__ == "__main__":
    main()

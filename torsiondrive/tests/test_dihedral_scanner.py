"""
Unit and regression test for the dihedral_scanner module.
"""

import pytest
import sys
import numpy as np
from torsiondrive.dihedral_scanner import DihedralScanner, Molecule
from torsiondrive.qm_engine import EngineBlank
from torsiondrive.priority_queue import PriorityQueue

def test_torsiondrive_imported():
    """Simple test, will always pass so long as import statement worked"""
    assert "torsiondrive" in sys.modules

def test_dihedral_scanner_setup():
    """
    Testing DihedralScanner Setup for 1-D to 4-D Scans
    """
    engine = EngineBlank()
    for dim in range(1, 5):
        print("Testing %d-D scan setup" % dim)
        dihedrals = [list(range(d, d+4)) for d in range(dim)]
        scanner = DihedralScanner(engine, dihedrals=dihedrals, grid_spacing=[90]*dim)
        gid = scanner.grid_ids[0]
        assert len(scanner.grid_ids) == 4**dim and len(gid) == dim, "Wrong dimension of grid_ids"
        assert len(scanner.grid_neighbors(gid)) == 2*dim, "Wrong dimension of grid_neighbors"
        assert isinstance(scanner.opt_queue, PriorityQueue), "DihedralScanner.opt_queue should be an instance of PriorityQueue"
        assert len(scanner.opt_queue) == 0, "DihedralScanner.opt_queue should be empty after setup"

def test_dihedral_scanner_methods():
    """
    Testing methods of DihedralScanner
    """
    # setup a scanner
    m = Molecule()
    m.elem = ['H'] * 5
    m.xyzs = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=float)*0.5]
    m.build_topology()
    engine = EngineBlank()
    dihedrals = [[0,1,2,3], [1,2,3,4]]
    scanner = DihedralScanner(engine, dihedrals=dihedrals, grid_spacing=[30, 30], init_coords_M=m)
    # test methods
    gid = (120, -60)
    assert scanner.get_dihedral_id(m) == gid
    assert scanner.get_dihedral_id(m, check_grid_id=(120, -90)) == None
    assert set(scanner.grid_neighbors(gid)) == {(90, -60), (150, -60), (120, -90), (120, -30)}
    assert set(scanner.grid_full_neighbors(gid)) == {(90, -90), (90, -30), (150, -90), (150, -30)}
    scanner.push_initial_opt_tasks()
    assert len(scanner.opt_queue) == 1
    geo, start_gid, end_gid  = scanner.opt_queue.pop()
    assert start_gid == end_gid

def test_dihedral_scanner_range_masks():
    """
    Test dihedral scanner range limit implemented as masks
    """
    # setup a scanner
    m = Molecule()
    m.elem = ['H'] * 5
    m.xyzs = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=float)*0.5]
    m.build_topology()
    engine = EngineBlank()
    dihedrals = [[0,1,2,3], [1,2,3,4]]
    # two ranges are tested, one is normal, one is "split" crossing boundaries
    dihedral_ranges = [[-120, 120], [150, 240]]
    scanner = DihedralScanner(engine, dihedrals=dihedrals, grid_spacing=[30, 30], init_coords_M=m, dihedral_ranges=dihedral_ranges)
    # check dihedral masks
    assert scanner.dihedral_ranges == dihedral_ranges
    assert scanner.dihedral_mask[0] == {-120, -90, -60, -30, 0, 30, 60, 90, 120}
    assert scanner.dihedral_mask[1] == {-150, -120, 150, 180}
    # test validate_task() function
    task1 = (m, (-120, 120), (-120, 150))
    assert scanner.validate_task(task1) == True
    task2 = (m, (-120, 60), (-120, 90))
    assert scanner.validate_task(task2) == False

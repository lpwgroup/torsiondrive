"""
Unit and regression test for the crank package.
"""

import pytest
import os, sys, subprocess, filecmp, shutil
import numpy as np
from crank.DihedralScanner import DihedralScanner, Molecule
from crank.QMEngine import QMEngine, EnginePsi4, EngineQChem, EngineTerachem
from crank.PriorityQueue import PriorityQueue

def test_crank_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "crank" in sys.modules

def test_dihedral_scanner_setup():
    """
    Testing DihedralScanner Setup for 1-D to 4-D Scans
    """
    engine = QMEngine()
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
    engine = QMEngine()
    dihedrals = [[0,1,2,3], [1,2,3,4]]
    scanner = DihedralScanner(engine, dihedrals=dihedrals, grid_spacing=[30, 30], init_coords_M=m)
    # test methods
    gid = (120, -60)
    assert scanner.get_dihedral_id(m) == gid
    assert set(scanner.grid_neighbors(gid)) == {(90, -60), (150, -60), (120, -90), (120, -30)}
    assert set(scanner.grid_full_neighbors(gid)) == {(90, -90), (90, -30), (150, -90), (150, -30)}
    scanner.push_initial_opt_tasks()
    assert len(scanner.opt_queue) == 1
    geo, start_gid, end_gid  = scanner.opt_queue.pop()
    assert start_gid == end_gid

def test_qm_engine():
    """
    Testing QMEngine Class
    """
    engine = QMEngine()
    assert hasattr(engine, 'temp_type')

def test_reproduce_1D_example():
    """
    Testing Reproducing Examples/hooh-1d
    """
    from crank import launch
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    example_path = os.path.join(this_file_folder, '..', '..', 'Examples')
    os.chdir(example_path)
    if not os.path.isdir('hooh-1d'):
        subprocess.call('tar zxf hooh-1d.tar.gz', shell=True)
    os.chdir('hooh-1d/psi4/run_local/geomeTRIC')
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs = launch.load_dihedralfile('dihedrals.txt')
    engine = launch.create_engine('psi4', inputfile='input.dat')
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')

def test_reproduce_2D_example():
    """
    Testing Reproducing Examples/propanol-2d
    """
    from crank import launch
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    example_path = os.path.join(this_file_folder, '..', '..', 'Examples')
    os.chdir(example_path)
    if not os.path.isdir('propanol-2d'):
        subprocess.call('tar zxf propanol-2d.tar.gz', shell=True)
    os.chdir('propanol-2d/work_queue_qchem_geomeTRIC')
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs = launch.load_dihedralfile('dihedrals.txt')
    engine = launch.create_engine('qchem', inputfile='qc.in')
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15, 15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')

def test_reproduce_api_example():
    """
    Testing Reproducing Examples/api_example
    """
    from crank import crankAPI
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    example_path = os.path.join(this_file_folder, '..', '..', 'Examples')
    os.chdir(example_path)
    if not os.path.isdir('api_example'):
        subprocess.call('tar zxf api_example.tar.gz', shell=True)
    os.chdir('api_example')
    current_state = crankAPI.current_state_json_load('current_state.json')
    next_jobs = crankAPI.get_next_jobs(current_state, verbose=True)
    crankAPI.next_jobs_json_dump(next_jobs, 'new.json')
    assert filecmp.cmp('new.json', 'next_jobs.json')

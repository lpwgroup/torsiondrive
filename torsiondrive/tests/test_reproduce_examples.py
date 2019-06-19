"""
Regression test for the torsiondrive package to reproduce the examples
"""

import pytest
import os
import sys
import subprocess
import shutil
import filecmp
import json
import numpy as np
from torsiondrive.dihedral_scanner import DihedralScanner

@pytest.fixture(scope="module")
def example_path(tmpdir_factory):
    """ Pytest fixture funtion to download the examples, decompress and return the path to the example/ folder """
    # tmpdir_factory is a pytest built-in fixture that has "session" scope
    tmpdir = tmpdir_factory.mktemp('torsiondrive_test_tmp')
    tmpdir.chdir()
    example_version = '0.9.8'
    url = f'https://github.com/lpwgroup/torsiondrive_examples/archive/v{example_version}.tar.gz'
    subprocess.run(f'wget -nc -q {url}', shell=True, check=True)
    subprocess.run(f'tar zxf v{example_version}.tar.gz', shell=True, check=True)
    os.chdir(f'torsiondrive_examples-{example_version}/examples')
    return os.getcwd()

def test_reproduce_1D_examples(example_path):
    """
    Testing Reproducing examples/hooh-1d
    """
    from torsiondrive import launch
    # reproduce psi4 local geomeTRIC
    os.chdir(example_path)
    os.chdir('hooh-1d/psi4/run_local/geomeTRIC')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs, dihedral_ranges = launch.load_dihedralfile('dihedrals.txt')
    engine = launch.create_engine('psi4', inputfile='input.dat')
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    # reproduce psi4 local native_opt
    os.chdir(example_path)
    os.chdir('hooh-1d/psi4/run_local/native_opt')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs, dihedral_ranges = launch.load_dihedralfile('dihedrals.txt')
    engine = launch.create_engine('psi4', inputfile='input.dat', native_opt=True)
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    # reproduce qchem local geomeTRIC
    os.chdir(example_path)
    os.chdir('hooh-1d/qchem/run_local/geomeTRIC')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs, dihedral_ranges = launch.load_dihedralfile('dihedrals.txt')
    engine = launch.create_engine('qchem', inputfile='qc.in')
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    # reproduce qchem local native_opt (skipped because qchem failed in this example)
    # os.chdir(example_path)
    # os.chdir('hooh-1d/qchem/run_local/native_opt')
    # subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    # shutil.copy('scan.xyz', 'orig_scan.xyz')
    # dihedral_idxs, dihedral_ranges = launch.load_dihedralfile('dihedrals.txt')
    # engine = launch.create_engine('qchem', inputfile='input.dat', native_opt=True)
    # scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    # scanner.master()
    # assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    # reproduce terachem local geomeTRIC
    os.chdir(example_path)
    os.chdir('hooh-1d/terachem/run_local/geomeTRIC')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs, dihedral_ranges = launch.load_dihedralfile('dihedrals.txt')
    engine = launch.create_engine('terachem', inputfile='run.in')
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    # reproduce terachem local native_opt
    os.chdir(example_path)
    os.chdir('hooh-1d/terachem/run_local/native_opt')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs, dihedral_ranges = launch.load_dihedralfile('dihedrals.txt')
    engine = launch.create_engine('terachem', inputfile='run.in', native_opt=True)
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    # reproduce openmm local geomeTRIC
    os.chdir(example_path)
    os.chdir('hooh-1d/openmm/run_local/geometric')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs, dihedral_ranges = launch.load_dihedralfile('dihedrals.txt')
    engine = launch.create_engine('openmm', inputfile='hooh.pdb')
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')

def test_reproduce_2D_example(example_path):
    """
    Testing Reproducing examples/propanol-2d
    """
    from torsiondrive import launch
    # reproduce psi4 work_queue geomeTRIC
    os.chdir(example_path)
    os.chdir('propanol-2d')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    argv = sys.argv[:]
    with open('run_command') as f:
        sys.argv = f.read().split()
    launch.main()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')

def test_reproduce_range_limit_example(example_path):
    """
    Testing Reproducing examples/range_limited
    """
    from torsiondrive import launch
    # reproduce qchem work_queue geomeTRIC
    os.chdir(example_path)
    os.chdir('range_limited')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    argv = sys.argv[:]
    with open('run_command') as f:
        sys.argv = f.read().split()
    launch.main()
    sys.argv = argv
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')

def test_reproduce_range_limit_split_example(example_path):
    """
    Testing Reproducing examples/range_limited_split
    """
    from torsiondrive import launch
    # reproduce qchem work_queue geomeTRIC
    os.chdir(example_path)
    os.chdir('range_limited')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    argv = sys.argv[:]
    with open('run_command') as f:
        sys.argv = f.read().split()
    launch.main()
    sys.argv = argv
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')

def test_reproduce_api_example(example_path):
    """
    Testing Reproducing examples/api_example
    """
    from torsiondrive import td_api
    # test running api
    os.chdir(example_path)
    os.chdir('api_example')
    orig_next_jobs = json.load(open('next_jobs.json'))
    current_state = json.load(open('current_state.json'))
    next_jobs = td_api.next_jobs_from_state(current_state, verbose=True)
    for grid_id, jobs in orig_next_jobs.items():
        assert np.allclose(jobs, next_jobs.get(grid_id, 0))
    # test writing current state
    loaded_state = td_api.current_state_json_load(current_state)
    td_api.current_state_json_dump(loaded_state, 'new_current_state.json')
    assert filecmp.cmp('current_state.json', 'new_current_state.json')
    # test calling td_api in command line
    shutil.copy('next_jobs.json', 'orig_next_jobs.json')
    with open('run_command') as f:
        sys.argv = f.read().split()
    td_api.main()
    assert filecmp.cmp('next_jobs.json', 'orig_next_jobs.json')

def test_reproduce_extra_constraints_example(example_path):
    """
    Testing Reproducing examples/extra_constraints
    """
    from torsiondrive import launch
    os.chdir(example_path)
    os.chdir('extra_constraints')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    argv = sys.argv[:]
    with open('run_command') as f:
        sys.argv = f.read().split()
    launch.main()
    sys.argv = argv
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')

def test_reproduce_energy_upper_limit_example(example_path):
    """
    Testing Reproducing examples/extra_constraints
    """
    from torsiondrive import launch
    os.chdir(example_path)
    os.chdir('energy_upper_limit')
    # reproduce ring example
    os.chdir('limit_ring_0.05')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    argv = sys.argv[:]
    with open('run_command') as f:
        sys.argv = f.read().split()
    launch.main()
    sys.argv = argv
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    os.chdir('..')
    # reproduce ring example
    os.chdir('limit_flexible_0.01')
    subprocess.run('tar zxf opt_tmp.tar.gz', shell=True, check=True)
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    argv = sys.argv[:]
    with open('run_command') as f:
        sys.argv = f.read().split()
    launch.main()
    sys.argv = argv
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')

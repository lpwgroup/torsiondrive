"""
Unit and regression test for the torsiondrive package.
"""

import pytest
import os, sys, subprocess, filecmp, shutil, json
import numpy as np
from torsiondrive.dihedral_scanner import DihedralScanner, Molecule
from torsiondrive.qm_engine import QMEngine, EnginePsi4, EngineQChem, EngineTerachem
from torsiondrive.priority_queue import PriorityQueue

def test_torsiondrive_imported():
    """Simple test, will always pass so long as import statement worked"""
    assert "torsiondrive" in sys.modules

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
    from torsiondrive.qm_engine import check_all_float
    assert check_all_float([1,0.2,3]) == True
    assert check_all_float([1,'a']) == False
    engine = QMEngine()
    with pytest.raises(NotImplementedError):
        engine.load_input('qc.in')
    engine.set_dihedral_constraints([[0,1,2,3,90]])
    engine.write_constraints_txt()
    assert os.path.isfile('constraints.txt')
    os.unlink('constraints.txt')
    engine.run('ls')
    assert engine.find_finished_jobs([], wait_time=1) == set()
    with pytest.raises(OSError):
        engine.load_task_result_m()
    assert engine.optimize_native() is None
    assert engine.optimize_geomeTRIC() is None
    assert engine.load_native_output() is None


def test_engine_psi4_native():
    """
    Testing EnginePsi4
    """
    os.mkdir('test.tmp')
    os.chdir('test.tmp')
    with open('input.dat', 'w') as psi4in:
        psi4in.write("""
molecule {
0 1
H  -1.116 -0.681 -0.191
O  -0.519  0.008 -0.566
O   0.518  0.074  0.561
H   1.126 -0.641  0.258
units angstrom
}
set basis 6-31g

optimize('mp2')
""")
    engine = EnginePsi4(input_file='input.dat', native_opt=True)
    assert hasattr(engine, 'M')
    engine.set_dihedral_constraints([[0,1,2,3,90]])
    try:
        engine.optimize_native()
        m = engine.load_native_output()
        assert pytest.approx(-150.9647, 0.0001) == m.qm_energies[0]
    except subprocess.CalledProcessError:
        pass
    os.chdir('..')
    shutil.rmtree('test.tmp')

def test_engine_psi4_geometric():
    """
    Testing EnginePsi4 by geomeTRIC
    """
    os.mkdir('test.tmp')
    os.chdir('test.tmp')
    with open('input.dat', 'w') as psi4in:
        psi4in.write("""
molecule {
0 1
H  -1.116 -0.681 -0.191
O  -0.519  0.008 -0.566
O   0.518  0.074  0.561
H   1.126 -0.641  0.258
units angstrom
}
set basis 6-31g

gradient('mp2')
""")
    engine = EnginePsi4(input_file='input.dat')
    assert hasattr(engine, 'M')
    engine.set_dihedral_constraints([[0,1,2,3,90]])
    try:
        engine.optimize_geomeTRIC()
        m = engine.load_geomeTRIC_output()
        assert pytest.approx(-150.9647, 0.0001) == m.qm_energies[0]
    except subprocess.CalledProcessError:
        pass
    os.chdir('..')
    shutil.rmtree('test.tmp')

def test_engine_qchem_native():
    """
    Testing EngineQChem
    """
    os.mkdir('test.tmp')
    os.chdir('test.tmp')
    with open('qc.in', 'w') as outfile:
        outfile.write("""
        $molecule
        0 1
        H  -3.20093  1.59945  -0.91132
        O  -2.89333  1.61677  -0.01202
        O  -1.41314  1.60154   0.01202
        H  -1.10554  1.61886   0.91132
        $end

        $rem
        jobtype              opt
        exchange             hf
        basis                3-21g
        geom_opt_max_cycles  150
        $end
        """)
    engine = EngineQChem(input_file='qc.in', native_opt=True)
    assert hasattr(engine, 'M')
    engine.set_dihedral_constraints([[0,1,2,3,90]])
    try:
        engine.optimize_native()
        m = engine.load_native_output()
        assert pytest.approx(-149.9420, 0.0001) == m.qm_energies[0]
    except subprocess.CalledProcessError:
        pass
    os.chdir('..')
    shutil.rmtree('test.tmp')

def test_engine_qchem_geometric():
    """
    Testing EngineQChem by geomeTRIC
    """
    os.mkdir('test.tmp')
    os.chdir('test.tmp')
    with open('qc.in', 'w') as outfile:
        outfile.write("""
        $molecule
        0 1
        H  -3.20093  1.59945  -0.91132
        O  -2.89333  1.61677  -0.01202
        O  -1.41314  1.60154   0.01202
        H  -1.10554  1.61886   0.91132
        $end

        $rem
        jobtype              force
        exchange             hf
        basis                3-21g
        geom_opt_max_cycles  150
        $end
        """)
    engine = EngineQChem(input_file='qc.in')
    assert hasattr(engine, 'M')
    engine.set_dihedral_constraints([[0,1,2,3,90]])
    try:
        engine.optimize_geomeTRIC()
        m = engine.load_geomeTRIC_output()
        assert pytest.approx(-149.9420, 0.0001) == m.qm_energies[0]
    except subprocess.CalledProcessError:
        pass
    os.chdir('..')
    shutil.rmtree('test.tmp')

def test_engine_terachem_native():
    """
    Testing EngineTerachem
    """
    os.mkdir('test.tmp')
    os.chdir('test.tmp')
    with open('run.in', 'w') as outfile:
        outfile.write("""
        coordinates start.xyz
        run minimize
        basis 6-31g*
        method rb3lyp
        charge 0
        spinmult 1
        dispersion yes
        scf diis+a
        maxit 50
        """)
    with open('start.xyz', 'w') as outfile:
        outfile.write("""4\n
        H  -3.20093  1.59945  -0.91132
        O  -2.89333  1.61677  -0.01202
        O  -1.41314  1.60154   0.01202
        H  -1.10554  1.61886   0.91132
        """)
    engine = EngineTerachem(input_file='run.in', native_opt=True)
    assert hasattr(engine, 'M')
    engine.set_dihedral_constraints([[0,1,2,3,90]])
    try:
        engine.optimize_native()
        m = engine.load_native_output()
        assert pytest.approx(-151.5334, 0.0001) == m.qm_energies[0]
    except subprocess.CalledProcessError:
        pass
    os.chdir('..')
    shutil.rmtree('test.tmp')

def test_engine_terachem_geometric():
    """
    Testing EngineTerachem by geomeTRIC
    """
    os.mkdir('test.tmp')
    os.chdir('test.tmp')
    with open('run.in', 'w') as outfile:
        outfile.write("""
        coordinates start.xyz
        run gradient
        basis 6-31g*
        method rb3lyp
        charge 0
        spinmult 1
        dispersion yes
        scf diis+a
        maxit 50
        """)
    with open('start.xyz', 'w') as outfile:
        outfile.write("""4\n
        H  -3.20093  1.59945  -0.91132
        O  -2.89333  1.61677  -0.01202
        O  -1.41314  1.60154   0.01202
        H  -1.10554  1.61886   0.91132
        """)
    engine = EngineTerachem(input_file='run.in')
    assert hasattr(engine, 'M')
    engine.set_dihedral_constraints([[0,1,2,3,90]])
    try:
        engine.optimize_geomeTRIC()
        m = engine.load_geomeTRIC_output()
        assert pytest.approx(-151.5334, 0.0001) == m.qm_energies[0]
    except subprocess.CalledProcessError:
        pass
    os.chdir('..')
    shutil.rmtree('test.tmp')

def test_reproduce_1D_examples():
    """
    Testing Reproducing examples/hooh-1d
    """
    from torsiondrive import launch
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    example_path = os.path.join(this_file_folder, '..', '..', 'examples')
    os.chdir(example_path)
    subprocess.call('tar zxf hooh-1d.tar.gz ', shell=True)
    # reproduce psi4 local geomeTRIC
    os.chdir('hooh-1d/psi4/run_local/geomeTRIC')
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs = launch.load_dihedralfile('dihedrals.txt', zero_based_numbering=True)
    engine = launch.create_engine('psi4', inputfile='input.dat')
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    os.chdir(example_path)
    # reproduce psi4 local native_opt
    os.chdir('hooh-1d/psi4/run_local/native_opt')
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs = launch.load_dihedralfile('dihedrals.txt', zero_based_numbering=True)
    engine = launch.create_engine('psi4', inputfile='input.dat', native_opt=True)
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    os.chdir(example_path)
    # reproduce qchem local geomeTRIC
    os.chdir('hooh-1d/qchem/run_local/geomeTRIC')
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs = launch.load_dihedralfile('dihedrals.txt', zero_based_numbering=True)
    engine = launch.create_engine('qchem', inputfile='qc.in')
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    os.chdir(example_path)
    # reproduce terachem local geomeTRIC
    os.chdir('hooh-1d/terachem/run_local/geomeTRIC')
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    dihedral_idxs = launch.load_dihedralfile('dihedrals.txt', zero_based_numbering=True)
    engine = launch.create_engine('terachem', inputfile='run.in')
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=[15], verbose=True)
    scanner.master()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')

def test_reproduce_2D_example():
    """
    Testing Reproducing examples/propanol-2d
    """
    from torsiondrive import launch
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    example_path = os.path.join(this_file_folder, '..', '..', 'examples')
    os.chdir(example_path)
    subprocess.call('tar zxf propanol-2d.tar.gz', shell=True)
    # reproduce qchem work_queue geomeTRIC
    os.chdir('propanol-2d/work_queue_qchem_geomeTRIC')
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    argv = sys.argv[:]
    sys.argv = 'torsiondrive-launch qc.in dihedrals.txt -e qchem -g 15 --zero_based_numbering -v'.split()
    launch.main()
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    os.chdir(example_path)
    # reproduce qchem work_queue native_opt
    os.chdir('propanol-2d/work_queue_qchem_native_opt')
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    shutil.copy('scan.xyz', 'orig_scan.xyz')
    sys.argv = 'torsiondrive-launch qc.in dihedrals.txt -e qchem -g 15 --native_opt --zero_based_numbering -v'.split()
    launch.main()
    sys.argv = argv
    assert filecmp.cmp('scan.xyz', 'orig_scan.xyz')
    os.chdir(example_path)

def test_reproduce_api_example():
    """
    Testing Reproducing examples/api_example
    """
    from torsiondrive import td_api
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    example_path = os.path.join(this_file_folder, '..', '..', 'examples')
    os.chdir(example_path)
    subprocess.call('tar zxf api_example.tar.gz', shell=True)
    # test running api
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
    os.chdir(example_path)

@pytest.mark.skipif("work_queue" not in sys.modules, reason='work_queue not found')
def test_work_queue():
    from torsiondrive.wq_tools import WorkQueue
    import time
    wq = WorkQueue(56789)
    wq.submit('echo test > test.txt', [], ['test.txt'])
    assert wq.get_queue_status() == (0,0,0,0)
    # submit a worker
    p = subprocess.Popen("$HOME/opt/cctools/bin/work_queue_worker localhost 56789 -t 1", shell=True)
    for _ in range(10):
        path = wq.check_finished_task_path()
        if path is not None:
            assert path == os.getcwd()
            break
    wq.print_queue_status()
    assert os.path.isfile('test.txt')
    assert open('test.txt').read().strip() == 'test'
    os.unlink('test.txt')

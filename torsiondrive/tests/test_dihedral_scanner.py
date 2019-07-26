"""
Unit and regression test for the dihedral_scanner module.
"""

import pytest
import sys
import numpy as np
from warnings import warn
from torsiondrive.dihedral_scanner import DihedralScanner, Molecule, measure_dihedrals
from torsiondrive.qm_engine import EngineBlank
from torsiondrive.priority_queue import PriorityQueue

def test_torsiondrive_imported():
    """Simple test, will always pass so long as import statement worked"""
    import torsiondrive
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

def test_dihedral_scanner_energy_upper_limit_filter():
    """
    Test dihedral scanner energy_upper_limit as task filters
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
    scanner = DihedralScanner(engine, dihedrals=dihedrals, grid_spacing=[30, 30], init_coords_M=m, dihedral_ranges=dihedral_ranges, energy_upper_limit=0.001)
    # check dihedral masks
    assert scanner.energy_upper_limit == 0.001
    # test validate_task() function
    scanner.global_minimum_energy = 0.0
    m.qm_energies = [0.0]
    task1 = (m, (-120, 120), (-120, 150))
    assert scanner.validate_task(task1) == True
    m.qm_energies = [0.0015]
    task2 = (m, (-120, 120), (-120, 150))
    assert scanner.validate_task(task2) == False

def test_measure_dihedrals():
    """
    Test measure_dihedrals function with test molecules
    """
    #  molecule 1; molecule containing four atoms lying in a straight line(CH3CCH)
    m1 = Molecule()
    m1.elem = ['C', 'C', 'H', 'C', 'H', 'H', 'H']
    m1.xyzs = [np.array([[-3.247,  2.208, -1.587],
                         [-2.045,  2.208, -1.587],
                         [-0.975,  2.208, -1.587],
                         [-4.787,  2.208, -1.587],
                         [-5.143,  2.681, -0.697],
                         [-5.143,  2.742, -2.443],
                         [-5.143,  1.2  , -1.623]])]
    # check if the function raises error when wrong dihedral_list is wrong
    with pytest.raises(AssertionError) as e_info:
        wrong_dihedrals1 = [[1,0,3]]
        measure_dihedrals(m1, wrong_dihedrals1, check_linear=False, check_bonded=False)
        assert len(e_enfo) != 0
    with pytest.raises(IndexError) as e_info:
        wrong_dihedrals2 = [[1,0,3,10]]
        measure_dihedrals(m1, wrong_dihedrals2, check_linear=False, check_bonded=False)
        assert len(e_enfo) != 0
    # check if measure_dihedrals raise a warning for dihedral containing a straight angle
    with pytest.warns(UserWarning) as record:
        dihedrals1 = [[1,0,3,4]]
        dihedral_values = measure_dihedrals(m1, dihedrals1, check_linear=True, check_bonded=True)
        assert len(record) == 1
        assert 'straight' in record[0].message.args[0].lower()
        # check if the dihedral_values have the same length with dihedrals and if the return value matches
        assert len(dihedral_values) == len(dihedrals1)
        assert dihedral_values == [0.0]
    # check if measure_dihedrals raise a warning for dihedral containing two atoms in the exactly same position
    with pytest.warns(UserWarning) as record:
        dihedrals2 = [[0,0,3,4]]
        dihedral_values = measure_dihedrals(m1, dihedrals2, check_linear=True, check_bonded=False)
        assert len(record) == 1
        assert 'same coordinate' in record[0].message.args[0].lower()
    # check if measure_dihedrals raise a warning for dihedral containing  a nonbonded atom sequence
    with pytest.warns(UserWarning) as record:
        dihedrals3 = [[2,1,0,4]]
        dihedral_values = measure_dihedrals(m1, dihedrals3, check_linear=False, check_bonded=True)
        assert len(record) == 1
        assert 'bond' in record[0].message.args[0].lower()
    # molecule 2; methanol
    m2 = Molecule()
    m2.elem = ['O', 'H', 'C', 'H', 'H', 'H']
    m2.xyzs = [np.array([[ 0.040, -0.730,  0.000],
                         [-0.856, -1.032,  0.000],
                         [ 0.061,  0.669, -0.000],
                         [ 1.098,  0.973, -0.000],
                         [-0.419,  1.084,  0.884],
                         [-0.419,  1.084, -0.884]])]
    dihedrals = [[1,0,2,3], [1,0,2,4], [1,0,2,5]]
    with pytest.warns(None) as record:
        dihedral_values = measure_dihedrals(m2, dihedrals, check_linear=True, check_bonded=True)
        assert not record.list
        assert len(dihedral_values) == len(dihedrals)
        answers = [180.0, 61.190466239931894, -61.190466239931894]
        np.testing.assert_allclose(dihedral_values, answers, atol=1e-6)
    # molecule 3; methanol with slightly rotated methyl group
    m3 = Molecule()
    m3.elem = ['O', 'H', 'C', 'H', 'H', 'H']
    m3.xyzs = [np.array([[ 0.040, -0.730,  0.000],
                         [-0.856, -1.032,  0.000],
                         [ 0.061,  0.669, -0.000],
                         [ 0.951,  0.975, -0.532],
                         [ 0.106,  1.076,  1.008],
                         [-0.805,  1.090, -0.507]])]
    with pytest.warns(None) as record:
        dihedral_values = measure_dihedrals(m3, dihedrals, check_linear=True, check_bonded=True)
        assert not record.list
        assert len(dihedral_values) == len(dihedrals)
        answers = [-148.997437219388, 92.2092401128753, -30.1683461569122]
        np.testing.assert_allclose(dihedral_values, answers, atol=1e-6)
    # molecule 4; C=C=C=C
    m4 = Molecule()
    m4.elem = ['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    m4.xyzs = [np.array([[ 1.0151,  0.2224, -0.0757],
                         [ 2.2176, -0.3464,  0.0653],
                         [ 2.8544, -1.1032, -0.9837],
                         [ 4.0567, -1.6731, -0.8433],
                         [ 0.5728,  0.7838,  0.7412],
                         [ 0.4435,  0.1469, -0.9953],
                         [ 2.7467, -0.2388,  1.0099],
                         [ 2.3255, -1.21  , -1.9286],
                         [ 4.6282, -1.5988,  0.0763],
                         [ 4.4985, -2.2344, -1.6606]])]
    dihedrals = [[0,1,2,3], [0,1,2,7], [6,1,2,7]]
    with pytest.warns(None) as record:
        dihedral_values = measure_dihedrals(m4, dihedrals, check_linear=True, check_bonded=True)
        assert not record.list
        assert len(dihedral_values) == len(dihedrals)
        answers = [-179.95906345380854, 0.04886624744276984, -179.94896540856553]
        np.testing.assert_allclose(dihedral_values, answers, atol=1e-6)
    # molecule 5; a simple ring compound, benzene
    m5 = Molecule()
    m5.elem = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    m5.xyzs = [np.array([[-7.6000e-01,  1.1691e+00, -5.0000e-04],
                         [ 6.3290e-01,  1.2447e+00, -1.2000e-03],
                         [ 1.3947e+00,  7.6500e-02,  4.0000e-04],
                         [ 7.6410e-01, -1.1677e+00,  2.7000e-03],
                         [-6.2880e-01, -1.2432e+00,  1.0000e-04],
                         [-1.3907e+00, -7.5100e-02, -1.5000e-03],
                         [-1.3536e+00,  2.0792e+00,  5.0000e-04],
                         [ 1.1243e+00,  2.2140e+00, -2.8000e-03],
                         [ 2.4799e+00,  1.3550e-01, -0.0000e+00],
                         [ 1.3576e+00, -2.0778e+00,  6.3000e-03],
                         [-1.1202e+00, -2.2126e+00, -5.0000e-04],
                         [-2.4759e+00, -1.3400e-01, -3.5000e-03]])]
    dihedrals = [[6,0,1,7], [5,0,1,7], [6,0,1,2], [5,0,1,2]]
    with pytest.warns(None) as record:
        dihedral_values = measure_dihedrals(m5, dihedrals, check_linear=True, check_bonded=True)
        assert not record.list
        assert len(dihedral_values) == len(dihedrals)
        answers = [-0.1250906915581075, 179.8551830096832, 179.86320790765137, -0.15651839110736443]
        np.testing.assert_allclose(dihedral_values, answers, atol=1e-6)

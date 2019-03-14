"""
Test for qm_engine module
"""

import os
import subprocess
import pytest
from torsiondrive.qm_engine import QMEngine, EngineBlank, EnginePsi4, EngineQChem, EngineTerachem

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
    with pytest.raises(NotImplementedError):
        engine.optimize_native()
    with pytest.raises(NotImplementedError):
        engine.optimize_geomeTRIC()
    with pytest.raises(NotImplementedError):
        engine.load_native_output()


def test_engine_blank():
    engine = EngineBlank()
    engine.optimize_native()
    engine.optimize_geomeTRIC()
    engine.load_native_output()


def test_engine_psi4_native(tmpdir):
    """
    Testing EnginePsi4
    """
    tmpdir.chdir()
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

def test_engine_psi4_geometric(tmpdir):
    """
    Testing EnginePsi4 by geomeTRIC
    """
    tmpdir.chdir()
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

def test_engine_qchem_native(tmpdir):
    """
    Testing EngineQChem
    """
    tmpdir.chdir()
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

def test_engine_qchem_geometric(tmpdir):
    """
    Testing EngineQChem by geomeTRIC
    """
    tmpdir.chdir()
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

def test_engine_terachem_native(tmpdir):
    """
    Testing EngineTerachem
    """
    tmpdir.chdir()
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

def test_engine_terachem_geometric(tmpdir):
    """
    Testing EngineTerachem by geomeTRIC
    """
    tmpdir.chdir()
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
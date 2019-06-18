"""
Test for qm_engine module
"""

import os
import subprocess
import pytest
from torsiondrive.qm_engine import QMEngine, EngineBlank, EnginePsi4, EngineQChem, EngineTerachem, EngineOpenMM

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

def test_engine_OpenMM_geometric(tmpdir):
    """
    Testing EngineOpenMM by geomeTRIC
    """
    tmpdir.chdir()
    with open('tdrive.pdb', 'w+') as pdb:
        pdb.write("""REMARK   1 CREATED WITH GEOMETRIC 2019-06-14
HETATM    1  O00 UNK     1       1.000   1.000   0.000  0.00  0.00           O
HETATM    2  O01 UNK     1      -0.453   1.000   0.000  0.00  0.00           O  
HETATM    3  H02 UNK     1       1.111   1.000  -0.970  0.00  0.00           H  
HETATM    4  H03 UNK     1      -0.564   0.999   0.970  0.00  0.00           H  
TER       5      UNK     1
CONECT    1    2    3
CONECT    2    1    4
CONECT    3    1
CONECT    4    2
""")
    with open('tdrive.xml', 'w+') as xml:
        xml.write("""<ForceField>
<AtomTypes>
<Type name="opls_802" class="H802" element="H" mass="1.008000" />
<Type name="opls_803" class="H803" element="H" mass="1.008000" />
<Type name="opls_800" class="O800" element="O" mass="15.999000" />
<Type name="opls_801" class="O801" element="O" mass="15.999000" />
</AtomTypes>
<Residues>
<Residue name="UNK">
<Atom name="O00" type="opls_800" />
<Atom name="O01" type="opls_801" />
<Atom name="H02" type="opls_802" />
<Atom name="H03" type="opls_803" />
<Bond from="0" to="1"/>
<Bond from="0" to="2"/>
<Bond from="1" to="3"/>
</Residue>
</Residues>
<HarmonicBondForce>
<Bond class1="O801" class2="O800" length="0.128000" k="454039.312000"/>
<Bond class1="H802" class2="O800" length="0.094500" k="462750.400000"/>
<Bond class1="H803" class2="O801" length="0.094500" k="462750.400000"/>
</HarmonicBondForce>
<HarmonicAngleForce>
<Angle class1="O801" class2="O800" class3="H802" angle="1.936094" k="390.367200"/>
<Angle class1="O800" class2="O801" class3="H803" angle="1.936094" k="390.367200"/>
</HarmonicAngleForce>
<PeriodicTorsionForce>
<Proper class1="H803" class2="O801" class3="O800" class4="H802" k1="0.000000" k2="0.000000" k3="0.736384" k4="0.000000" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
</PeriodicTorsionForce>
<NonbondedForce coulomb14scale="0.5" lj14scale="0.5" combination="opls">
<Atom type="opls_802" charge="0.428000" sigma="0.000000" epsilon="0.000000" />
<Atom type="opls_803" charge="0.428000" sigma="0.000000" epsilon="0.000000" />
<Atom type="opls_800" charge="-0.428000" sigma="0.312000" epsilon="0.711280" />
<Atom type="opls_801" charge="-0.428000" sigma="0.312000" epsilon="0.711280" />
</NonbondedForce>
</ForceField>
""")
    engine = EngineOpenMM(input_file='tdrive.pdb')
    assert hasattr(engine, 'M')
    engine.set_dihedral_constraints([[0,1,2,3,90]])
    try:
        engine.optimize_geomeTRIC()
        m = engine.load_geomeTRIC_output()
        assert pytest.approx(0.0205234, 0.00001) == m.qm_energies[0]
    except subprocess.CalledProcessError:
        pass

def test_engine_OpenMM_xml_missing(tmpdir):
    """
    Testing running OpenMM with no xml, expect AssertionError
    """
    tmpdir.chdir()
    with open('tdrive.pdb', 'w+') as pdb:
        pdb.write("""REMARK   1 CREATED WITH GEOMETRIC 2019-06-14
HETATM    1  O00 UNK     1       1.000   1.000   0.000  0.00  0.00           O
HETATM    2  O01 UNK     1      -0.453   1.000   0.000  0.00  0.00           O  
HETATM    3  H02 UNK     1       1.111   1.000  -0.970  0.00  0.00           H  
HETATM    4  H03 UNK     1      -0.564   0.999   0.970  0.00  0.00           H  
TER       5      UNK     1
CONECT    1    2    3
CONECT    2    1    4
CONECT    3    1
CONECT    4    2
""")
    with pytest.raises(AssertionError):
        engine = EngineOpenMM(input_file='tdrive.pdb')


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
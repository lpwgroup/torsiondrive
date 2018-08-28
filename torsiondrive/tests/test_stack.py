"""
Unit and regression test for the torsiondrive package.
Test the stack of torsiondrive -> geomeTRIC -> qcengine -> Psi4
"""

import pytest
import os, sys, json
import shutil
import numpy as np
from torsiondrive.dihedral_scanner import DihedralScanner, Molecule
from torsiondrive.qm_engine import QMEngine
import geometric
from geometric.nifty import bohr2ang

try:
    import qcengine
    import psi4
except ImportError:
    pass


class Psi4QCEngineEngine(QMEngine):
    def __init__(self, *args, **kwargs):
        super(Psi4QCEngineEngine, self).__init__(*args, **kwargs)
        self.stored_results = dict()

    def load_input(self, input_file):
        """ Load molecule from an xyz file """
        self.M = Molecule(input_file)

    def optimize_geomeTRIC(self):
        """ run the constrained optimization using geomeTRIC package:
        1. Create a json input for a constrained optimization
        2. Run the job
        """
        # step 1
        in_json_dict = self.create_in_json_dict()
        # step 2
        key = os.getcwd()
        self.stored_results[key] = geometric.run_json.geometric_run_json(in_json_dict)

    def create_in_json_dict(self):
        constraints_dict = {
                'set': [('dihedral', str(d1+1), str(d2+1), str(d3+1), str(d4+1), str(v)) for d1, d2, d3, d4, v in self.dihedral_idx_values]
        }
        qc_schema_input = {
            "schema_name": "qc_schema_input",
            "schema_version": 1,
            "driver": "gradient",
            "model": {
                "method": "MP2",
                "basis": "cc-pvdz"
            },
            "keywords": {}
        }
        in_json_dict = {
            "schema_name": "qc_schema_optimization_input",
            "schema_version": 1,
            "keywords": {
                "coordsys": "tric",
                'constraints': constraints_dict,
                "program": "psi4"
            },
            "initial_molecule": {
                "geometry": (self.M.xyzs[0] / bohr2ang).ravel().tolist(),
                "symbols": self.M.elem,
            },
            "input_specification": qc_schema_input
        }
        return in_json_dict

    def load_geomeTRIC_output(self):
        """ Load the optimized geometry and energy from self.out_json_dict into a new molecule object and return """
        out_json_dict = self.stored_results.pop(os.getcwd())
        mdict = out_json_dict['final_molecule']
        m = Molecule()
        m.xyzs = [np.array(mdict['geometry']).reshape(-1, 3) * bohr2ang]
        m.elem = mdict['symbols']
        m.build_topology()
        m.qm_energies = [out_json_dict['trajectory'][-1]['properties']['return_energy']]
        return m


@pytest.mark.skipif("qcengine" not in sys.modules, reason='qcengine not found')
@pytest.mark.skipif("psi4" not in sys.modules, reason='psi4 not found')
def test_stack_psi4():
    """
    Test the stack of torsiondrive -> geomeTRIC -> qcengine -> Psi4
    """
    orig_path = os.getcwd()
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files', 'hooh-psi4')
    os.chdir(test_folder)

    # Make sure to delete subfolders
    opt_tmp_folder = os.path.join(test_folder, "opt_tmp")
    shutil.rmtree(opt_tmp_folder, ignore_errors=True)

    engine = Psi4QCEngineEngine('start.xyz')
    scanner = DihedralScanner(engine, dihedrals=[[0, 1, 2, 3]], grid_spacing=[90], verbose=True)
    scanner.master()
    result_energies = [scanner.grid_energies[grid_id] for grid_id in sorted(scanner.grid_energies.keys())]
    assert np.allclose(
        result_energies, [-151.17367357, -151.16167615, -151.17367357, -151.17370632],
        atol=1e-4)
    os.chdir(orig_path)

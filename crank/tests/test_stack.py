"""
Unit and regression test for the crank package.
"""

import pytest
import os, sys, subprocess, filecmp, shutil, json
import numpy as np
from crank.DihedralScanner import DihedralScanner, Molecule
from crank.QMEngine import QMEngine
from crank.PriorityQueue import PriorityQueue
import geometric

try:
    import qcengine
    import psi4
except:
    pass

bohr2ang = 0.529177210

class Psi4QCEngineEngine(QMEngine):
    def __init__(self, *args, **kwargs):
        super(Psi4QCEngineEngine, self).__init__(*args, **kwargs)
        self.stored_results = dict()

    def load_input(self, input_file):
        """ Load molecule from an xyz file """
        self.M = Molecule(input_file)

    def optimize_geomeTRIC(self):
        """ run the constrained optimization using geomeTRIC package, in 3 steps:
        1. Write a constraints.txt file.
        2. Create a json input for a constrained optimization
        3. Run the job
        """
        # step 1
        self.write_constraints_txt()
        # step 2
        in_json_dict = self.create_in_json_dict()
        # step 3
        key = os.getcwd()
        self.stored_results[key] = geometric.run_json.geometric_run_json(in_json_dict)

    def create_in_json_dict(self):
        qc_schema_input = {
            "schema_name": "qc_schema_input",
            "schema_version": 1,
            "molecule": {
                "geometry": (self.M.xyzs[0]/bohr2ang).ravel().tolist(),
                "symbols": self.M.elem,
                "connectivity": [[int(i), int(j), 1] for i,j in self.M.bonds]
            },
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
                'constraints': 'constraints.txt',
                "program": "psi4"
            },
            "input_specification": qc_schema_input
        }
        return in_json_dict

    def load_geomeTRIC_output(self):
        """ Load the optimized geometry and energy from self.out_json_dict into a new molecule object and return """
        out_json_dict = self.stored_results.pop(os.getcwd())
        mdict = out_json_dict['final_molecule']
        m = Molecule()
        m.xyzs = [np.array(mdict['molecule']['geometry']).reshape(-1,3) * bohr2ang]
        m.elem = mdict['molecule']['symbols']
        m.build_topology()
        m.qm_energies = [mdict['properties']['return_energy']]
        return m

@pytest.mark.skipif("qcengine" not in sys.modules, reason='qcengine not found')
@pytest.mark.skipif("psi4" not in sys.modules, reason='psi4 not found')
def test_stack_psi4():
    """
    Test the stack of crank -> geomeTRIC -> qcengine -> Psi4
    """
    orig_path = os.getcwd()
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files', 'hooh-psi4')
    os.chdir(test_folder)
    engine = Psi4QCEngineEngine('start.xyz')
    #scanner = DihedralScanner(engine, dihedrals=[[0,1,6,10],[1,6,10,11]], grid_spacing=[15, 15], verbose=True)
    scanner = DihedralScanner(engine, dihedrals=[[0,1,2,3]], grid_spacing=[15], verbose=True)
    scanner.master()
    result_energies = [scanner.grid_energies[grid_id] for grid_id in sorted(scanner.grid_energies.keys())]
    assert np.allclose(result_energies, [-151.17383,-151.17416,-151.17455,-151.17477,-151.17455,-151.17367,-151.17199,
        -151.16962,-151.16686,-151.16424,-151.16236,-151.16167,-151.16236,-151.16424,-151.16686,-151.16962,-151.17199,
        -151.17367,-151.17455,-151.17477,-151.17455,-151.17416,-151.17383,-151.17370], atol=1e-4)
    os.chdir(orig_path)

# We commented out the RDKit tests now since it has several issues
# 1. The gradient is not good enough that geomeTRIC always struggle to find a good step
# 2. The bond orders are not determined, but needed by RDKit connectivity.

# try:
#     import qcengine
#     import rdkit
# except:
#     pass
#
# class RDkitEngine(QMEngine):
#     def __init__(self, *args, **kwargs):
#         super(RDkitEngine, self).__init__(*args, **kwargs)
#         self.stored_results = dict()
#
#     def load_input(self, input_file):
#         """ Load molecule from an xyz file """
#         self.M = Molecule(input_file)
#
#     def optimize_geomeTRIC(self):
#         """ run the constrained optimization using geomeTRIC package, in 3 steps:
#         1. Write a constraints.txt file.
#         2. Create a json input for a constrained optimization
#         3. Run the job
#         """
#         # step 1
#         self.write_constraints_txt()
#         # step 2
#         in_json_dict = self.create_in_json_dict()
#         # step 3
#         key = os.getcwd()
#         self.stored_results[key] = geometric.run_json.geometric_run_json(in_json_dict)
#
#     def create_in_json_dict(self):
#         qc_schema_input = {
#             "schema_name": "qc_schema_input",
#             "schema_version": 1,
#             "molecule": {
#                 "geometry": (self.M.xyzs[0]/bohr2ang).ravel().tolist(),
#                 "symbols": self.M.elem,
#                 "connectivity": [[int(i), int(j), 1] for i,j in self.M.bonds]
#             },
#             "driver": "gradient",
#             "model": {
#                 "method": "UFF",
#                 "basis": None
#             },
#             "keywords": {}
#         }
#         in_json_dict = {
#             "schema_name": "qc_schema_optimization_input",
#             "schema_version": 1,
#             "keywords": {
#                 "coordsys": "tric",
#                 'constraints': 'constraints.txt',
#                 "program": "rdkit"
#             },
#             "input_specification": qc_schema_input
#         }
#         return in_json_dict
#
#     def load_geomeTRIC_output(self):
#         """ Load the optimized geometry and energy from self.out_json_dict into a new molecule object and return """
#         out_json_dict = self.stored_results.pop(os.getcwd())
#         mdict = out_json_dict['final_molecule']
#         m = Molecule()
#         m.xyzs = [np.array(mdict['molecule']['geometry']).reshape(-1,3) * bohr2ang]
#         m.elem = mdict['molecule']['symbols']
#         m.build_topology()
#         m.qm_energies = [mdict['properties']['return_energy']]
#         return m
#
# @pytest.mark.skipif("qcengine" not in sys.modules, reason='qcengine not found')
# @pytest.mark.skipif("rdkit" not in sys.modules, reason='rdkit not found')
# def test_stack_RDKit():
#     """
#     Test the stack of crank -> geomeTRIC -> qcengine -> RDKit
#     """
#     orig_path = os.getcwd()
#     this_file_folder = os.path.dirname(os.path.realpath(__file__))
#     test_folder = os.path.join(this_file_folder, 'files', 'hooh')
#     os.chdir(test_folder)
#     engine = RDkitEngine('start.xyz')
#     #scanner = DihedralScanner(engine, dihedrals=[[0,1,6,10],[1,6,10,11]], grid_spacing=[15, 15], verbose=True)
#     scanner = DihedralScanner(engine, dihedrals=[[0,1,2,3]], grid_spacing=[15], verbose=True)
#     scanner.master()
#     os.chdir(orig_path)

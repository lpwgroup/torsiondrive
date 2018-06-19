"""
Unit and regression test for the crank package.
"""

import pytest
import os, sys, copy, collections
import numpy as np
from crank.DihedralScanner import Molecule
from crank.QMEngine import QMEngine
from crank.crankAPI import crank_api
import geometric

try:
    import psi4
except:
    pass

bohr2ang = 0.529177210

class FakeServer:
    """
    A fake server, that do these things:
    1. Take an xyz file and dihedrals list, create json input for crank
    2. Send the json input dictionary to crankAPI.crank_api(), get the next set of jobs
    3. Take the set of jobs and run them with engine. Finish if there're no new jobs.
    4. Collect the results and put them into new json input dictionary
    5. Go back to step 2.
    """
    def __init__(self, xyzfilename, dihedrals, grid_spacing):
        self.M = Molecule(xyzfilename)
        self.dihedrals = dihedrals
        self.grid_spacing = grid_spacing
        self.engine = Psi4QCEngineEngine()
        self.engine.M = copy.deepcopy(self.M)

    def run(self):
        # step 1
        in_dict = self.create_initial_api_input()
        while True:
            # step 2
            next_jobs = crank_api(in_dict, verbose=True)
            # step 3
            if len(next_jobs) > 0:
                # step 4
                job_results = self.run_jobs(next_jobs)
                in_dict = self.update_api_input(in_dict, job_results)
            else:
                print("Finished")
                return self.collect_lowest_energies(in_dict)

    def create_initial_api_input(self):
        """ Create the initial input dictionary for crank-api """
        return {
            'dihedrals': self.dihedrals,
            'grid_spacing': self.grid_spacing,
            'elements': self.M.elem,
            'init_coords': [self.M.xyzs[0].ravel().tolist()],
            'grid_status': {}
        }

    def run_jobs(self, next_jobs):
        """ Take a dictionary of next jobs, run them and return the results """
        job_results = collections.defaultdict(list)
        for grid_id_str, job_geo_list in next_jobs.items():
            grid_id = tuple(int(i) for i in grid_id_str.split(','))
            dihedral_idx_values = []
            for dihedral_idxs, dihedral_value in zip(self.dihedrals, grid_id):
                dihedral_idx_values.append(list(dihedral_idxs) + [dihedral_value])
            # running jobs in serial here, if run in parallel the order should be kept same
            for job_geo in job_geo_list:
                # run the job using engine
                self.engine.M.xyzs = [np.array(job_geo, dtype=float).reshape(-1, 3)]
                self.engine.set_dihedral_constraints(dihedral_idx_values)
                self.engine.optimize_geomeTRIC()
                m = self.engine.load_geomeTRIC_output()
                # here we check if the dihedrals after optimization matches the constraint
                # this is good for validity if the engine is not checking that
                if "final dihedral may be different than constraint":
                    dihedral_values = np.array([m.measure_dihedrals(*d)[0] for d in self.dihedrals])
                    for dv, dref in zip(dihedral_values, grid_id):
                        diff = abs(dv - dref)
                        if min(diff, abs(360-diff)) > 0.9:
                            print("Warning! dihedral values inconsistent with check_grid_id")
                            print('dihedral_values', dihedral_values, 'ref_grid_id', grid_id)
                    dihedral_id = (np.round(dihedral_values / self.grid_spacing) * self.grid_spacing).astype(int)
                    result_grid_id = [(d + (180-d)//360*360) for d in dihedral_id]
                else:
                    result_grid_id = grid_id
                # put the results into job_results dict, note that the order is same as input
                result_grid_id_str = ','.join(map(str, result_grid_id))
                start_geo = job_geo
                end_geo = m.xyzs[0].ravel().tolist()
                end_energy = m.qm_energies[0]
                job_result_tuple = (start_geo, end_geo, end_energy)
                job_results[result_grid_id_str].append(job_result_tuple)
        return job_results

    def update_api_input(self, in_dict, job_results):
        updated_dict = copy.deepcopy(in_dict)
        updated_dict['grid_status'] = collections.defaultdict(list, updated_dict['grid_status'])
        for grid_id_str, job_result_tuple_list in job_results.items():
            updated_dict['grid_status'][grid_id_str] += job_result_tuple_list
        return updated_dict

    def collect_lowest_energies(self, in_dict):
        lowest_energies = collections.defaultdict(lambda: float('inf'))
        for grid_id_str, job_result_tuple_list in in_dict['grid_status'].items():
            grid_id = tuple(int(i) for i in grid_id_str.split(','))
            for start_geo, end_geo, end_energy in job_result_tuple_list:
                lowest_energies[grid_id] = min(lowest_energies[grid_id], end_energy)
        return lowest_energies


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
def test_stack_fakeserver():
    """
    Test the stack of crank -> geomeTRIC -> qcengine -> Psi4
    """
    orig_path = os.getcwd()
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files', 'hooh-fakeserver')
    os.chdir(test_folder)
    fakeserver = FakeServer('start.xyz', dihedrals=[[0,1,2,3]], grid_spacing=[30])
    lowest_energies = fakeserver.run()
    result_energies = [lowest_energies[grid_id] for grid_id in sorted(lowest_energies.keys())]
    assert np.allclose(result_energies, [-151.17383,-151.17416,-151.17455,-151.17477,-151.17455,-151.17367,-151.17199,
        -151.16962,-151.16686,-151.16424,-151.16236,-151.16167,-151.16236,-151.16424,-151.16686,-151.16962,-151.17199,
        -151.17367,-151.17455,-151.17477,-151.17455,-151.17416,-151.17383,-151.17370][1::2], atol=1e-4)
    os.chdir(orig_path)

"""
Unit and regression test for the torsiondrive package.
Test the stack of (server <=> td_api) -> geomeTRIC -> qcengine -> Psi4
"""

import os, sys, collections, pytest
import numpy as np
import geometric
from geometric.nifty import ang2bohr

from torsiondrive import td_api
from torsiondrive.extra_constraints import check_conflict_constraints

try:
    import qcengine
    import psi4
except ImportError:
    pass


class SimpleServer:
    """ A simple server that interfaces with torsiondrive and geometric to do the dihedral scanning work flow """

    def __init__(self, xyzfilename, dihedrals, grid_spacing, dihedral_ranges=None, energy_decrease_thresh=None, energy_upper_limit=None, extra_constraints=None):
        self.M = geometric.molecule.Molecule(xyzfilename)
        self.dihedrals = dihedrals
        self.grid_spacing = grid_spacing
        self.elements = self.M.elem
        self.init_coords = [(self.M.xyzs[0] * ang2bohr).ravel().tolist()]
        self.dihedral_ranges = dihedral_ranges
        self.energy_decrease_thresh = energy_decrease_thresh
        self.energy_upper_limit = energy_upper_limit
        if extra_constraints is not None:
            check_conflict_constraints(extra_constraints, dihedrals)
        self.extra_constraints = extra_constraints

    def run_torsiondrive_scan(self):
        """
        Run torsiondrive scan in the following steps:
        1. Create json input for torsiondrive
        2. Send the json input dictionary to td_api.next_jobs_from_state(), get the next set of jobs
        3. If there are no jobs needed, finish and return the lowest energy on each dihedral grid
        4. If there are new jobs, run them with geomeTRIC.run_json.
        5. Collect the results and put them into new json input dictionary
        6. Go back to step 2.
        """

        # step 1
        td_state = td_api.create_initial_state(
            dihedrals=self.dihedrals,
            grid_spacing=self.grid_spacing,
            elements=self.elements,
            init_coords=self.init_coords,
            dihedral_ranges=self.dihedral_ranges,
            energy_decrease_thresh=self.energy_decrease_thresh,
            energy_upper_limit=self.energy_upper_limit,
        )

        while True:
            # step 2
            next_jobs = td_api.next_jobs_from_state(td_state, verbose=True)

            # step 3
            if len(next_jobs) == 0:
                print("torsiondrive Scan Finished")
                return td_api.collect_lowest_energies(td_state)

            # step 4
            job_results = collections.defaultdict(list)
            for grid_id_str, job_geo_list in next_jobs.items():
                for job_geo in job_geo_list:
                    dihedral_values = td_api.grid_id_from_string(grid_id_str)

                    # Run geometric
                    geometric_input_dict = self.make_geomeTRIC_input(dihedral_values, job_geo)
                    geometric_output_dict = geometric.run_json.geometric_run_json(geometric_input_dict)

                    # Pull out relevevant data
                    final_result = geometric_output_dict['trajectory'][-1]
                    final_geo = final_result['molecule']['geometry']
                    final_energy = final_result['properties']['return_energy']

                    # Note: the results should be appended in the same order as in the inputs
                    # It's not a problem here when running serial for loop
                    job_results[grid_id_str].append((job_geo, final_geo, final_energy))

            # step 5
            td_api.update_state(td_state, job_results)

    def make_geomeTRIC_input(self, dihedral_values, geometry):
        """ This function should be implemented on the server, that takes QM specs, geometry and constraint
        to generate a geomeTRIC json input dictionary"""
        constraints_dict = {
            'set': [{'type': 'dihedral', 'indices': list(d), 'value': v} for d, v in zip(self.dihedrals, dihedral_values)]
        }
        # merge the extra constraints in
        if self.extra_constraints is not None:
            for key, v_list in self.extra_constraints.items():
                constraints_dict.setdefault(key, [])
                constraints_dict[key] += v_list
        qc_schema_input = {
            "schema_name": "qcschema_input",
            "schema_version": 1,
            "driver": "gradient",
            "model": {
                "method": "SCF",
                "basis": "STO-3G"
            },
            "keywords": {}
        }

        geometric_input_dict = {
            "schema_name": "qcschema_optimization_input",
            "schema_version": 1,
            "keywords": {
                "coordsys": "tric",
                'constraints': constraints_dict,
                "program": "psi4"
            },
            "initial_molecule": {
                "geometry": geometry,
                "symbols": self.elements
            },
            "input_specification": qc_schema_input
        }

        return geometric_input_dict


@pytest.mark.skipif("qcengine" not in sys.modules, reason='qcengine not found')
@pytest.mark.skipif("psi4" not in sys.modules, reason='psi4 not found')
def test_stack_simpleserver():
    """
    Test the stack of (server <=> td_api) -> geomeTRIC -> qcengine -> Psi4
    """
    orig_path = os.getcwd()

    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files', 'hooh-simpleserver')
    os.chdir(test_folder)

    simpleServer = SimpleServer('start.xyz', dihedrals=[[0, 1, 2, 3]], grid_spacing=[60])
    lowest_energies = simpleServer.run_torsiondrive_scan()

    result_energies = [lowest_energies[grid_id] for grid_id in sorted(lowest_energies.keys())]
    assert np.allclose(
        result_energies, [-148.76511761, -148.76018225, -148.7505629, -148.76018225, -148.76511761, -148.76501337],
        atol=1e-4)

    os.chdir(orig_path)


@pytest.mark.skipif("qcengine" not in sys.modules, reason='qcengine not found')
@pytest.mark.skipif("psi4" not in sys.modules, reason='psi4 not found')
def test_stack_simpleserver_optional():
    """
    Test the stack of (server <=> td_api) -> geomeTRIC -> qcengine -> Psi4 with optional arguments
    """
    orig_path = os.getcwd()

    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files', 'hooh-simpleserver')
    os.chdir(test_folder)

    extra_constraints = {
        'freeze': [{
            "type": "distance",
            "indices": [0, 1],
        }]
    }
    simpleServer = SimpleServer('start.xyz', dihedrals=[[0, 1, 2, 3]], grid_spacing=[30], dihedral_ranges=[[-150, -60]], extra_constraints=extra_constraints)
    lowest_energies = simpleServer.run_torsiondrive_scan()

    result_energies = [lowest_energies[grid_id] for grid_id in sorted(lowest_energies.keys())]
    print(result_energies)
    assert np.allclose(
        result_energies, [-148.76400359, -148.76395751, -148.76286594, -148.75885779],
        atol=1e-4)

    os.chdir(orig_path)
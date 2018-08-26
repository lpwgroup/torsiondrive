#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import json
import os
import shutil
from collections import defaultdict

import numpy as np
from torsiondrive.dihedral_scanner import DihedralScanner, get_geo_key
from torsiondrive.priority_queue import PriorityQueue
from torsiondrive.qm_engine import QMEngine
from geometric.molecule import Molecule
from geometric.nifty import bohr2ang, ang2bohr

class DihedralScanRepeater(DihedralScanner):
    """ Child class of dihedral scanner, that is specifically designed to accommodate
    the requirements of torsiondrive-API"""
    def repeat_scan_process(self):
        """ Mimicing DihedralScanner.master function, but stops when new jobs needs to run """
        self.push_initial_opt_tasks()
        if len(self.opt_queue) == 0:
            print("No tasks in opt_queue! Exiting..")
            return
        # make sure we're in the rootpath
        os.chdir(self.rootpath)
        self.refined_grid_ids = set()
        self.next_jobs = defaultdict(list)
        self.current_finished_job_results = PriorityQueue()
        # start the iteration from beginning
        while True:
            # print current status
            if self.verbose:
                if len(self.dihedrals) == 2:
                    print(self.draw_ramachandran_plot())
                else:
                    print(self.draw_ascii_image())
            # this function will try to read cache and decide if new jobs needs to run
            self.launch_opt_jobs()
            # Break if any job was not found in the current cache
            if len(self.next_jobs) > 0: break
            # If all jobs found in the current iteration, parse the results
            current_best_grid_m = {}
            while len(self.current_finished_job_results) > 0:
                m, grid_id = self.current_finished_job_results.pop()
                if grid_id not in current_best_grid_m or m.qm_energies[0] < current_best_grid_m[grid_id].qm_energies[0]:
                    current_best_grid_m[grid_id] = m
            # we only want refined results in current iteration to show in draw_ramachandran_plot()
            self.refined_grid_ids = set()
            # compare the best results between current iteration and all previous iterations
            newly_updated_grid_m = []
            for grid_id, m in current_best_grid_m.items():
                if grid_id not in self.grid_energies:
                    if self.verbose:
                        print("First energy for grid_id %s = %f" % (str(grid_id), m.qm_energies[0]))
                    self.grid_energies[grid_id] = m.qm_energies[0]
                    self.grid_final_geometries[grid_id] = m.xyzs[0]
                    newly_updated_grid_m.append((grid_id, m))
                elif m.qm_energies[0] < self.grid_energies[grid_id] - self.energy_decrease_thresh:
                    if self.verbose:
                        print("Energy for grid_id %s decreased from %f to %f" % (str(grid_id), self.grid_energies[grid_id],
                                                                                 m.qm_energies[0]))
                    self.grid_energies[grid_id] = m.qm_energies[0]
                    self.grid_final_geometries[grid_id] = m.xyzs[0]
                    newly_updated_grid_m.append((grid_id, m))
                    # we record the refined_grid_ids here to be printed as green tiles in draw_ramachandran_plot()
                    self.refined_grid_ids.add(grid_id)
            # create new tasks for each newly_updated_grid_m
            for grid_id, m in newly_updated_grid_m:
                # every neighbor grid point will get one new task
                for neighbor_gid in self.grid_neighbors(grid_id):
                    task = m, grid_id, neighbor_gid
                    # all jobs are pushed with the same priority for now, can be adjusted here
                    self.opt_queue.push(task)
            # check if all jobs finished
            if len(self.opt_queue) == 0 and len(self.next_jobs) == 0:
                print("All optimizations converged at lowest energy. Job Finished!")
                break

    def rebuild_task_cache(self, grid_status):
        """
        Take a dictionary of finished optimizations, rebuild task_cache dictionary
        This function mimics the DihedralScanner.restore_task_cache()

        Parameters:
        ------------
        grid_status = dict(), key is the grid_id, value is a list of job_info. Each job_info is a tuple of (start_geo, end_geo, end_energy).
            * Note: The order of the job_info is important when reproducing the same scan procedure.

        Returns: None
        ------------
        Upon finish, the new folder 'opt_tmp' will be created, with many empty folders corrsponding to the finished jobs.
        self.task_cache will be populated with correct information for repreducing the entire scan process.
        """
        for grid_id, job_info_list in grid_status.items():
            tname = 'gid_' + '_'.join('%+04d' % gid for gid in grid_id)
            tmp_folder_path = os.path.join(self.tmp_folder_name, tname)
            for i_job, job_info in enumerate(job_info_list):
                job_path = os.path.join(tmp_folder_path, str(i_job + 1))
                (start_geo, end_geo, end_energy) = job_info
                job_geo_key = get_geo_key(start_geo)
                self.task_cache[grid_id][job_geo_key] = (end_geo, end_energy, job_path)

    def launch_opt_jobs(self):
        """
        Mimicing DihedralScanner.launch_opt_jobs,
        """
        assert hasattr(self, 'next_jobs') and hasattr(self, 'current_finished_job_results')
        while len(self.opt_queue) > 0:
            m, from_grid_id, to_grid_id = self.opt_queue.pop()
            # check if this job already exists
            m_geo_key = get_geo_key(m.xyzs[0])
            if m_geo_key in self.task_cache[to_grid_id]:
                final_geo, final_energy, job_folder = self.task_cache[to_grid_id][m_geo_key]
                result_m = Molecule()
                result_m.elem = list(m.elem)
                result_m.xyzs = [final_geo]
                result_m.qm_energies = [final_energy]
                result_m.build_topology()
                grid_id = self.get_dihedral_id(result_m, check_grid_id=to_grid_id)
                self.current_finished_job_results.push((result_m, grid_id), priority=job_folder)
            else:
                # append the job to self.next_jobs, which is the output of torsiondrive-API
                self.next_jobs[to_grid_id].append(m.xyzs[0].copy())


def get_next_jobs(current_state, verbose=False):
    """
    Take current scan state and generate the next set of optimizations.
    This function will create a new DihedralScanRepeater object and read all information from current_state,
    then reproduce the entire scan from the beginning, finish all cached ones, until a new job is not found in the cache.
    Return a list of new jobs that needs to be finished for the current iteration

    Input:
    -------
    current_state: dict, e.g. {
            'dihedrals': [[0,1,2,3], [1,2,3,4]] ,
            'grid_spacing': [30, 30],
            'elements': ['H', 'C', 'O', ...]
            'init_coords': [geo1, geo2, ..]
            'grid_status': {(30, 60): [(start_geo, end_geo, end_energy), ..], ...}
        }

    Output:
    -------
    next_jobs: dict(), key is the target grid_id, value is a list of new_job. Each new_job is represented by its start_geo
        * Note: the order of new_job should correspond to the finished job_info.

    """
    dihedrals = current_state['dihedrals']
    grid_spacing = current_state['grid_spacing']
    # rebuild the init_coords_M molecule object
    init_coords_M = Molecule()
    init_coords_M.elem = current_state['elements']
    init_coords_M.xyzs = current_state['init_coords']
    init_coords_M.build_topology()
    # create a new scanner object
    scanner = DihedralScanRepeater(QMEngine(), dihedrals, grid_spacing, init_coords_M, verbose)
    # rebuild the task_cache for scanner
    scanner.rebuild_task_cache(current_state['grid_status'])
    # run the scanner until some calculation is not found in cache
    scanner.repeat_scan_process()
    return scanner.next_jobs


def current_state_json_dump(current_state, jsonfilename):
    """ Dump a state to a JSON file """
    json_state = current_state.copy()
    json_state['init_coords'] = [(c * ang2bohr).ravel().tolist() for c in current_state['init_coords']]
    json_state['grid_status'] = dict()

    for grid_id, grid_jobs in current_state['grid_status'].items():
        grid_id_str = ','.join(map(str, grid_id))
        new_grid_jobs = []
        for start_geo, end_geo, end_energy in grid_jobs:
            new_grid_jobs.append([(start_geo * ang2bohr).ravel().tolist(), (end_geo * ang2bohr).ravel().tolist(),
                                  end_energy])
        json_state['grid_status'][grid_id_str] = new_grid_jobs

    with open(jsonfilename, 'w') as outfile:
        json.dump(json_state, outfile, indent=2)


def current_state_json_load(json_state_dict):
    """ Load a state from JSON dictionary """

    json_state = copy.deepcopy(json_state_dict)
    natoms = len(json_state['elements'])

    # convert geometries into correct numpy format
    init_coords = [np.array(c, dtype=float).reshape(natoms, 3) * bohr2ang for c in json_state['init_coords']]
    json_state['init_coords'] = init_coords

    # convert grid_status into dictionary
    grid_status = defaultdict(list)

    # create a molecule object here to evaluate dihedrals later
    m = Molecule()
    m.xyzs = init_coords
    m.elem = json_state['elements']
    m.build_bonds()

    dihedrals = json_state['dihedrals']
    grid_spacing = json_state['grid_spacing']
    # create grid status dictionary
    for grid_id_str, grid_jobs in json_state['grid_status'].items():
        grid_id = tuple(int(i) for i in grid_id_str.split(','))
        for start_geo, end_geo, end_energy in grid_jobs:

            # convert to numpy array, shape should match here
            start_geo = np.array(start_geo, dtype=float).reshape(natoms, 3) * bohr2ang
            end_geo = np.array(end_geo, dtype=float).reshape(natoms, 3) * bohr2ang

            # here we check if the end_geo matches the target grid id
            m.xyzs = [end_geo]
            dihedral_values = np.array([m.measure_dihedrals(*d)[0] for d in dihedrals])
            for dv, dref in zip(dihedral_values, grid_id):
                diff = abs(dv - dref)
                if min(diff, abs(360 - diff)) > 0.9:
                    print("Warning! dihedral values inconsistent with target grid_id")
                    print('dihedral_values', dihedral_values, 'ref_grid_id', grid_id)

            dihedral_id = (np.round(dihedral_values / grid_spacing) * grid_spacing).astype(int)
            real_grid_id = tuple((d + (180 - d) // 360 * 360) for d in dihedral_id)

            # here we append the result into the real grid_id
            grid_status[real_grid_id].append((start_geo, end_geo, end_energy))

    json_state['grid_status'] = grid_status
    return json_state


def next_jobs_json_dict(next_jobs):
    """ Dump the next_jobs dictionary to a json file """
    json_next_jobs = {}
    for grid_id, new_job_list in next_jobs.items():
        grid_id_str = ','.join(map(str, grid_id))
        json_job_list = [(new_job_geo * ang2bohr).ravel().tolist() for new_job_geo in new_job_list]
        json_next_jobs[grid_id_str] = json_job_list
    return json_next_jobs


def next_jobs_from_state(td_state, verbose=False):
    """ The main function of torsiondrive API, which takes a torsiondrive state in JSON dictionary,
    and returns a JSON dictionary with information of next jobs.

    Parameters
    ----------
    td_state : dict
        A dictionary description of the torsiondrive state
    verbose : bool, optional
        Extra printing or not.

    Returns
    -------
    dict
        A dictionary of jobs to run.
        If the entire torsiondrive scan has finished, will return an empty dictionary
    """
    current_state = current_state_json_load(td_state)
    next_jobs = get_next_jobs(current_state, verbose=verbose)
    json_next_jobs = next_jobs_json_dict(next_jobs)
    return json_next_jobs


### Utility functions for servers


def create_initial_state(dihedrals, grid_spacing, elements, init_coords):
    """Create the initial input dictionary for torsiondrive API

    Parameters
    ----------
    dihedrals : list of tuples
        A list of the dihedrals to scan over.
    grid_spacing : list of int
        The grid seperation for each dihedral angle
    elements : list of strings
        Symbols for all elements in the molecule
    init_coords : list of (N, 3) or (N*3) arrays
        The initial coordinates in bohr

    Returns
    -------
    dict
        A representation of the torsiondrive state as JSON
    """
    return {
        'dihedrals': dihedrals,
        'grid_spacing': grid_spacing,
        'elements': elements,
        'init_coords': init_coords,
        'grid_status': {}
    }


def collect_lowest_energies(td_state):
    """
    Find the lowest energies for each dihedral grid from td_state
    """
    lowest_energies = defaultdict(lambda: float('inf'))
    for grid_id_str, job_result_tuple_list in td_state['grid_status'].items():
        grid_id = grid_id_from_string(grid_id_str)
        for start_geo, end_geo, end_energy in job_result_tuple_list:
            lowest_energies[grid_id] = min(lowest_energies[grid_id], end_energy)

    # Must convert back to standard dictionary
    return dict(lowest_energies)


def update_state(td_state, job_results):
    """
    Updates the torsiondrive state with the compute jobs. The state is updated inplace

    Parameters
    ----------
    td_state : dict
        The current torsiondrive state
    job_results : dict
        A dictionary of completed jobs and job ID's

    Returns
    -------
    None
    """
    for grid_id_str, job_result_tuple_list in job_results.items():
        if grid_id_str not in td_state['grid_status']:
            td_state['grid_status'][grid_id_str] = []
        td_state['grid_status'][grid_id_str] += job_result_tuple_list
    return td_state


def grid_id_from_string(grid_id_str):
    """Convert

    Parameters
    ----------
    grid_id_str : str
        The string grid ID representation

    Returns
    -------
    ret : tuple of ints
        A 4-length tuple representation of the dihedral id
    """
    return tuple(int(i) for i in grid_id_str.split(','))

### End of Utility functions used by server


def main():
    import argparse, sys
    parser = argparse.ArgumentParser(
        description="Take a scan state and return the next set of optimizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('statefile', help='File contains the current state in JSON format')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False, help='Print more information while running.')
    args = parser.parse_args()

    # print input command for reproducibility
    print(' '.join(sys.argv))

    # Load json dictionary from file
    json_dict = json.load(open(args.statefile))

    # run the api
    json_next_jobs = next_jobs_from_state(json_dict, verbose=args.verbose)

    # dump results to file
    json.dump(json_next_jobs, open('next_jobs.json', 'w'), indent=2)
    print("Information for next set of jobs is dumped to next_jobs.json")

    # print results
    if len(json_next_jobs) > 0:
        print("Number of jobs to run next for each grid id")
        for grid_id in json_next_jobs.keys():
            print("%-20s %10d" % (str(grid_id), len(json_next_jobs[grid_id])))
    else:
        print("All torsiondrive jobs finished.")


if __name__ == "__main__":
    main()

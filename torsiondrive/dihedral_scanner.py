#!/usr/bin/env python
# -*- coding: utf-8 -*-

#=================================================#
#| Minimum-energy multi-dimensional torsion scan |#
#|         Yudong Qiu, Lee-Ping Wang             |#
#================================================#

from __future__ import print_function, division

import collections
import copy
import itertools
import json
import os
import pickle
import time

import numpy as np
from geometric.molecule import Molecule
from torsiondrive.priority_queue import PriorityQueue


def normalize_dihedral(d):
    """ Normalize any number to the range (-180, 180], including 180 """
    return d + (180-d)//360*360

def get_geo_key(coords):
    """
    Convert an numpy array of xyz coordinate to a hashable object, keeping 0.001 precision
    This function has the limitation that 3.1999 and 3.2000 will produce different results
    due to the limitation of float point representation.
    """
    return (coords * 1000).astype(int).tobytes()

class DihedralScanner:
    """
    DihedralScanner class is designed to create a dihedral grid, and fill in optimized geometries and energies
    into the grid, by running wavefront propagations of constrained optimizations
    """
    def __init__(self, engine, dihedrals, grid_spacing, init_coords_M=None, verbose=False):
        """
        inputs:
        -------
        engine: An QMEngine object, e.g. EnginePsi4, EngineQChem or EngineTerachem
        dihedrals: list of dihedral index tuples (d1, d2, d3, d4). The length of list determines the dimension of the grid
                i.e. dihedrals = [(0,1,2,3)] --> 1-D scan,  dihedrals = [(0,1,2,3),(1,2,3,4)] --> 2-D Scan
        grid_spacing: Distance (in Degrees) between grid points, correspond to each dihedral, every value must be a divisor of 360
        init_coords_M: a geometric.molecule.Molecule object, constains a series of initial geometries to start with
        verbose: let methods print more information when running
        """
        self.engine = engine
        # validate input dihedral format
        self.dihedrals = []
        for dihedral in dihedrals:
            assert len(dihedral) == 4, "each dihedral in dihedrals should have 4 indices, e.g. (1,2,3,4)"
            dihedral_tuple = tuple(map(int, dihedral))
            assert dihedral_tuple not in self.dihedrals, "All dihedrals should be unique"
            self.dihedrals.append(dihedral_tuple)
        self.grid_dim = len(self.dihedrals)
        for gs in grid_spacing:
            assert (0 < gs < 360) and (360 % gs == 0), "grid_spacing %s is not valid, all values should be a divisor of 360" % grid_spacing
        assert len(grid_spacing) == self.grid_dim, "Number of grid spacings %d is not consistent with number of dihedrals %d" % (len(grid_spacing), self.grid_dim)
        self.grid_spacing = tuple(map(int, grid_spacing))
        self.setup_grid()
        self.opt_queue = PriorityQueue()
        # try to use init_coords_M first, if not given, use M in engine's template
        # `for m in init_coords_M` doesn't work since m.measure_dihedrals will fail because it has different m.xyzs shape
        self.init_coords_M = [init_coords_M[i] for i in range(len(init_coords_M))] if init_coords_M is not None else [self.engine.M]
        self.verbose = verbose
        # dictionary that stores the lowest energy for each grid point
        self.grid_energies = dict()
        # dictionary that stores the geometries corresponding to lowest energy for each grid point
        self.grid_final_geometries = dict()
        # save current path as the rootpath
        self.rootpath = self.engine.rootpath = os.getcwd()
        # path for temporary optimization files to be saved
        self.tmp_folder_name = 'opt_tmp'
        # task cache for restoring
        self.task_cache = collections.defaultdict(dict)
        # filename for storing finished task result
        self.task_result_fname = 'dihedral_scanner_task_result.p'
        # threshold for determining the energy decrease
        self.energy_decrease_thresh = 0.00001

    #----------------------
    # Initializing methods
    #----------------------

    def setup_grid(self):
        """
        Set up grid ids, each as a tuple with size corresponding to grid dimension. i.e.
        1-D: grid_ids = ( (-165, ), (-150, ), ... (180, )  )
        2-D: grid_ids = ( (-165,-165), (-165,-150), ... (180,180)  )
        This function is called by the initializer.
        """
        self.grid_axes = []
        for gs in self.grid_spacing:
            self.grid_axes.append(range(-180+gs, 180+gs, gs))
        self.grid_ids = tuple(itertools.product(*self.grid_axes))

    #--------------------
    #  General methods
    #--------------------
    def get_dihedral_id(self, molecule, check_grid_id=None):
        """
        Compute the closest grid ID for molecule (only first frame)
        If check_grid_id is given, will perform a check if the computed dihedral_values are close to the grid_id provided
        """
        dihedral_values = np.array([molecule.measure_dihedrals(*d)[0] for d in self.dihedrals])
        if check_grid_id is not None:
            assert len(check_grid_id) == len(dihedral_values), "Grid dimensions should be the same!"
            for dv, dref in zip(dihedral_values, check_grid_id):
                diff = abs(dv - dref)
                if min(diff, abs(360-diff)) > 0.9:
                    print("Warning! dihedral values inconsistent with check_grid_id")
                    print('dihedral_values', dihedral_values, 'check_grid_id', check_grid_id)
                    break
        dihedral_id = (np.round(dihedral_values / self.grid_spacing) * self.grid_spacing).astype(int)
        # we return a tuples as the grid_id
        return tuple([normalize_dihedral(d) for d in dihedral_id])

    def grid_neighbors(self, grid_id):
        """ Take a center grid id, return all the neighboring grid ids, in each dimension """
        neighbor_gridids = []
        for i_dim in range(len(grid_id)):
            gs = self.grid_spacing[i_dim]
            lower_neighbor = list(grid_id)
            lower_neighbor[i_dim] = normalize_dihedral(grid_id[i_dim] - gs)
            neighbor_gridids.append(tuple(lower_neighbor))
            higher_neighbor = list(grid_id)
            higher_neighbor[i_dim] = normalize_dihedral(grid_id[i_dim] + gs)
            neighbor_gridids.append(tuple(higher_neighbor))
        return tuple(neighbor_gridids)

    def grid_full_neighbors(self, grid_id):
        """ Take a center grid id, return all the neighboring grid ids, in all dimensions """
        # Note: This function is not in use now, because it's very expensive (and probably unnecessary)
        neighbor_gids_each_dim = []
        for gid_each_dim, gs in zip(grid_id, self.grid_spacing):
            lower_neighbor = normalize_dihedral(gid_each_dim - gs)
            higher_neighbor = normalize_dihedral(gid_each_dim + gs)
            neighbor_gids_each_dim.append((lower_neighbor, higher_neighbor))
        return tuple(itertools.product(*neighbor_gids_each_dim))

    #----------------------
    # Master method
    #----------------------
    def master(self):
        """
        The master function that calls all other functions.
        This function will run the following steps:
        1. Launch a new set of jobs from self.opt_queue, add their job path to a dictionary
        2. Check if any running job has finished
        3. For each finished job, check if energy is lower than existing one, if so, add its neighbor grid points to opt_queue
        4. Go back to the 1st step, loop until all jobs finished, indicated by opt_queue and running jobs both empty.
        """
        print("Master process started!")
        self.push_initial_opt_tasks()
        if len(self.opt_queue) == 0:
            print("No tasks in opt_queue! Exiting..")
            return
        # make sure we're in the rootpath
        os.chdir(self.rootpath)
        # check if the tmp folder exists
        if os.path.isdir(self.tmp_folder_name):
            # use existing tmp folder and read task cache
            self.restore_task_cache()
        else:
            # setup new tmp folders
            self.create_tmp_folder()
        # dictionary that saves the information for each running job, like orig m, orig grid_id, target grid_id
        self.running_job_path_info = dict()
        # Queue that saves the finished job results for each iteration
        # In each iteration, this will be populated by job results from task cache, and from new calculations
        # After parsing the finished jobs, this will emptied for the next iteration
        # We used a PriorityQueue here so the order of parsing finished jobs will be kept
        self.current_finished_job_results = PriorityQueue()
        # print scan status interval
        start_time = last_print_time = time.time()
        # the minimum time interval between prints
        min_print_interval = -1 # Disabled for now
        # store the grid ids that have found lower energy than existing one, for draw_ramachandran_plot()
        self.refined_grid_ids = set()
        # save the status of grid from beginning of run, useful when generating state files
        # self.grid_status = collections.defaultdict(list)
        while True:
            # check if it's time to show the status
            current_time = time.time()
            if self.verbose and current_time - last_print_time > min_print_interval:
                print("Scan Status at %d s" % (current_time-start_time))
                if len(self.dihedrals) == 2:
                    print(self.draw_ramachandran_plot())
                else:
                    print(self.draw_ascii_image())
                last_print_time = current_time
            # Launch all jobs in self.opt_queue
            # new jobs will be put into self.running_job_path_info
            # job results found in self.task_cache will be added to self.current_finished_job_results
            self.launch_opt_jobs()
            # wait until all jobs finish, take out from self.running_job_path_info
            while len(self.running_job_path_info) > 0:
                self.wait_extract_finished_jobs()
            # check all finished jobs and keep the best ones for the current iteration
            current_best_grid_m = dict()
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
                        print("Energy for grid_id %s decreased from %f to %f" % (str(grid_id), self.grid_energies[grid_id], m.qm_energies[0]))
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
            if len(self.opt_queue) == 0 and len(self.running_job_path_info) == 0:
                print("All optimizations converged at lowest energy. Job Finished!")
                break
        # the finish function will write files like scan.xyz, qdata.txt to disk
        self.finish()


    #----------------------------------
    # Utility methods Called by Master
    #----------------------------------

    def push_initial_opt_tasks(self):
        """
        Push a set of initial tasks to self.opt_queue
        A task is defined as (m, from_grid_id, to_grid_id) tuple, where geometry is stored in m
        """
        for m in self.init_coords_M:
            from_grid_id = to_grid_id = self.get_dihedral_id(m)
            task = (m, from_grid_id, to_grid_id)
            self.opt_queue.push(task)
        if self.verbose:
            print("%d initial tasks pushed to opt_queue" % len(self.init_coords_M))

    def save_task_cache(self, job_path, m_init, m_final, final_energy):
        """
        Save a file containing the finished job information to a pickle file on disk.
        The format should be consistent with self.restore_task_cache()
        """
        task_result = {'initial_geo': m_init.xyzs[0], 'final_geo': m_final.xyzs[0], 'final_energy': final_energy}
        with open(os.path.join(self.rootpath, job_path, self.task_result_fname), 'wb') as pickleout:
            pickle.dump(task_result, pickleout)


    def restore_task_cache(self):
        """
        Restore previous finished tasks from tmp folder.
        1. Look into tmp folder and read scanner_settings.json, check if it matches current setting
        2. Read the result pickle file from each leaf folder, into task_cache
        If successful, self.tmp_folder_dict will be initialized, same as self.create_tmp_folder(),
        and self.task_cache will be populated, with task caches, defined in this way:
            self.task_cache = {(30,-60): {geo_key: (final_geo, final_energy)}}
        """
        if self.verbose:
            print("Restoring from %s" % self.tmp_folder_name)
        # check if this scan matches the previous scan
        settings_fname = os.path.join(self.tmp_folder_name, 'scanner_settings.json')
        with open(settings_fname) as jsonfile:
            scanner_settings = json.load(jsonfile)
        err_msg = "Previous job doesn't match current one, please delete %s to restart" % self.tmp_folder_name
        assert len(self.dihedrals) == len(scanner_settings['dihedrals']), err_msg
        assert np.array_equal(np.array(self.dihedrals), np.array(scanner_settings['dihedrals'])), err_msg
        assert np.array_equal(self.grid_spacing, scanner_settings['grid_spacing']), err_msg
        # read all finished jobs in tmp folder
        self.tmp_folder_dict = dict()
        n_cache = 0
        for grid_id in self.grid_ids:
            tname = 'gid_' + '_'.join('%+04d' % gid for gid in grid_id)
            tmp_folder_path = os.path.join(self.tmp_folder_name, tname)
            self.tmp_folder_dict[grid_id] = tmp_folder_path
            existing_job_folders = [os.path.join(tmp_folder_path, f) for f in os.listdir(tmp_folder_path)]
            for job_folder in existing_job_folders:
                result_fname = os.path.join(job_folder, self.task_result_fname)
                if os.path.isfile(result_fname):
                    try:
                        task_result = pickle.load(open(result_fname, 'rb'))
                        task_geo_key = get_geo_key(task_result['initial_geo'])
                        self.task_cache[grid_id][task_geo_key] = (task_result['final_geo'], task_result['final_energy'], job_folder)
                        n_cache += 1
                    except Exception as e:
                        print("Error while loading result_fname:" + str(e))
                        pass
        if self.verbose:
            print("Successfully loaded %s cached results" % n_cache)

    def create_tmp_folder(self):
        """
        Create an empty tmp folder structure, save the paths for each grid point into self.tmp_folder_dict
        Example:
            self.tmp_folder_dict = {(30,-70): "opt_tmp/gid_+030_-070", ..}
        """
        assert hasattr(self, 'grid_ids'), 'Call self.setup_grid() first'
        os.mkdir(self.tmp_folder_name)
        # save current scan settings
        scanner_settings = {'dihedrals': self.dihedrals, 'grid_spacing': self.grid_spacing}
        settings_fname = os.path.join(self.rootpath, self.tmp_folder_name, 'scanner_settings.json')
        with open(settings_fname, 'w') as jsonfile:
            json.dump(scanner_settings, jsonfile)
        # create folders and save their path to self.tmp_folder_dict
        tmp_folder_dict = dict()
        for grid_id in self.grid_ids:
            tname = 'gid_' + '_'.join('%+04d' % gid for gid in grid_id)
            tmp_folder_path = os.path.join(self.tmp_folder_name, tname)
            os.mkdir(tmp_folder_path)
            tmp_folder_dict[grid_id] = tmp_folder_path
        self.tmp_folder_dict = tmp_folder_dict

    def launch_opt_jobs(self):
        """
        Launch constrained optimizations for molecules in opt_queue
        The current opt_queue will be cleaned up
        Return a dictionary that contains path and grid_ids: { path: (from_grid_id, to_grid_id) }
        """
        assert hasattr(self, 'running_job_path_info') and hasattr(self, 'current_finished_job_results')
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
                #self.grid_status[to_grid_id].append((m.xyzs[0], final_geo, final_energy))
            else:
                job_path = self.launch_constrained_opt(m, to_grid_id)
                self.running_job_path_info[job_path] = m, from_grid_id, to_grid_id

    def launch_constrained_opt(self, molecule, grid_id):
        """
        Called by launch_opt_jobs() to launch one opt job in a new scr folder
        Return the new folder path
        """
        dihedral_idx_values = []
        for dihedral_idxs, dihedral_value in zip(self.dihedrals, grid_id):
            dihedral_idx_values.append(list(dihedral_idxs) + [dihedral_value])
        # get a new folder
        new_job_path = self.get_new_scr_folder(grid_id)
        if self.verbose:
            print("Launching new job at %s" % new_job_path)
        # launch optimization job inside scratch folder
        self.engine.M = copy.deepcopy(molecule)
        self.engine.set_dihedral_constraints(dihedral_idx_values)
        self.engine.launch_optimize(new_job_path)
        return new_job_path

    def get_new_scr_folder(self, grid_id):
        """
        create a job scratch folder inside tmp_folder_dict[grid_id]
        name starting from '1', and will use larger numbers if exist
        return the new folder name that's been created
        """
        tmp_path = self.tmp_folder_dict[grid_id]
        existing_jobs = os.listdir(tmp_path)
        next_job_number = len(existing_jobs) + 1
        job_path = os.path.join(tmp_path, str(next_job_number))
        os.mkdir(job_path)
        return job_path

    def wait_extract_finished_jobs(self):
        """
        Interface with engine to check if any job finished
        Will wait infinitely here until at least one job finished
        The finished job paths will be removed from self.running_job_path_info
        The finished job results (m, grid_id) will be added to self.current_finished_job_results
        """
        if len(self.running_job_path_info) == 0:
            print("No job running, returning")
            return
        while True:
            finished_path_set = self.engine.find_finished_jobs(self.running_job_path_info, wait_time=3)
            if len(finished_path_set) > 0:
                break
        if self.verbose:
            print("Find finished jobs:", finished_path_set)
        for job_path in finished_path_set:
            m_init, from_grid_id, to_grid_id = self.running_job_path_info.pop(job_path)
            # call the engine to parse output file and return final geometry/energy in a new molecule
            m = self.engine.load_task_result_m(job_path)
            # we will check here if the optimized structure has the desired dihedral ids
            grid_id = self.get_dihedral_id(m, check_grid_id=to_grid_id)
            # save the parsed task result to disk
            self.save_task_cache(job_path, m_init, m, m.qm_energies[0])
            # each finished job result is a tuple of (m, grid_id)
            self.current_finished_job_results.push((m, grid_id), priority=job_path)

    def finish(self):
        """ Write qdata.txt and scan.xyz file based on converged scan results """
        m = Molecule()
        m.elem = list(self.engine.M.elem)
        m.qm_energies, m.xyzs, m.comms = [], [], []
        for gid in self.grid_ids:
            m.qm_energies.append(self.grid_energies[gid])
            m.xyzs.append(self.grid_final_geometries[gid])
            m.comms.append("Dihedral %s Energy %.9f" % (str(gid), self.grid_energies[gid]))
        m.write('qdata.txt')
        print("Final scan energies are written to qdata.txt")
        m.write('scan.xyz')
        print("Final scan energies are written to scan.xyz")


    #----------------------------------
    # Status Drawing Utilites
    #----------------------------------

    def draw_ascii_image(self):
        """ Return a string with ASCII colors showing current running status """
        if not hasattr(self, 'grid_energies') or not hasattr(self, 'opt_queue'):
            return "draw_ascii_image failed: grid_energies or opt_queue not available"
        result_str = ""
        count = 0
        running_to_job_ids = set(to_grid_id for m, from_grid_id, to_grid_id in self.opt_queue)
        for grid_id in self.grid_ids:
            symbol = ' -'
            if grid_id in running_to_job_ids:
                symbol = ' \033[0;33m+\033[0m' # orange for running jobs
            elif grid_id in self.grid_energies:
                symbol = ' \033[0;36mo\033[0m' # cyan for finished jobs
            result_str += symbol
            count += 1
            end_number = 1
            for i_dim in range(self.grid_dim):
                end_number *= len(self.grid_axes[i_dim])
                if count % end_number == 0:
                    result_str += '\n'
        return result_str

    def draw_ramachandran_plot(self):
        """ Return a string of Ramachandran plot showing current running status """
        assert self.grid_dim == 2, "Ramachandran plot only works for 2-D scans"
        gsx, gsy = self.grid_spacing
        grid_x, grid_y = self.grid_axes
        # add labels of status for each grid point
        grid_status = collections.defaultdict(str)
        gid_direction = {(gsx,0):'r', (gsx-360,0):'r', (-gsx,0):'l', (360-gsx,0):'l',
                         (0,gsy):'u', (0,gsy-360):'u', (0,-gsy):'d', (0,360-gsy):'d', (0,0):'o'}
        # print the status of jobs that are about to be launched
        for m, from_grid_id, to_grid_id in self.opt_queue:
            from_x, from_y = from_grid_id
            to_x, to_y = to_grid_id
            direction = gid_direction[(to_x-from_x, to_y-from_y)]
            grid_status[to_grid_id] += direction
        # if no job launching for this grid point, print the previous result
        for grid_id in self.refined_grid_ids:
            if grid_id not in grid_status:
                grid_status[grid_id] = 'f' # green tiles for just refined results
        for grid_id in self.grid_energies:
            if grid_id not in grid_status:
                grid_status[grid_id] = 'e' # blue tiles for finished results
        # format string
        status_symbols = collections.defaultdict(lambda: '\x1b[1;41m><\x1b[0m',
                           {''  :'  '                  , 'c': '\x1b[46m--\x1b[0m',
                            'e' :'\x1b[44m--\x1b[0m'   , 'f': '\x1b[42m--\x1b[0m',
                            'r' :'\x1b[1;41m＞\x1b[0m' , 'l':'\x1b[1;41m＜\x1b[0m',
                            'd' :'\x1b[1;41m\\/\x1b[0m', 'u':'\x1b[1;41m/\\\x1b[0m',
                            'dl':'\x1b[41m＼\x1b[0m'   , 'dr':'\x1b[41m／\x1b[0m',
                            'ld':'\x1b[41m＼\x1b[0m'   , 'rd':'\x1b[41m／\x1b[0m',
                            'ul':'\x1b[41m／\x1b[0m'   , 'ur':'\x1b[41m＼\x1b[0m',
                            'lu':'\x1b[41m／\x1b[0m'   , 'ru':'\x1b[41m＼\x1b[0m',
                           })
        result_str  = "--== Ramachandran Plot of Optimization Status ==--\n"
        result_str += "--== Blue: Optimized, Green: Found Lower, Red: Next ==--\n"
        result_str += "  " + ''.join("%6d" % x for x in grid_x[::3]) + '\n'
        for y in grid_y[::-1]:
            line = '%4d '%y + ''.join(status_symbols[grid_status[(x,y)]] for x in grid_x) + '\n'
            result_str += line
        return result_str

#!/usr/bin/env python
# -*- coding: utf-8 -*-

#=================================================#
#| Minimum-energy multi-dimensional torsion scan |#
#|         Yudong Qiu, Lee-Ping Wang             |#
#================================================#

from __future__ import print_function, division
import numpy as np
import os, shutil, time, itertools
from forcebalance.molecule import Molecule
from QMEngine import EnginePsi4, EngineQChem, EngineTerachem
from PriorityQueue import PriorityQueue

class DihedralScanner:
    """
    DihedralScanner class is designed to create a dihedral grid, and fill in optimized geometries and energies
    into the grid, by running wavefront propagations of constrained optimizations
    """
    def __init__(self, engine, dihedrals, grid_spacing=15, init_coords_M=None, verbose=False):
        """
        inputs:
        -------
        engine: An QMEngine object, e.g. EnginePsi4, EngineQChem or EngineTerachem
        dihedrals: list of dihedral index tuples (d1, d2, d3, d4). The length of list determines the dimension of the grid
                i.e. dihedrals = [(0,1,2,3)] --> 1-D scan,  dihedrals = [(0,1,2,3),(1,2,3,4)] --> 2-D Scan
        grid_spacing: Distance (in Degrees) between grid points, must be a divisor of 360
        init_coords_M: a forcebalance.molecule.Molecule object, constains a series of initial geometries to start with
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
        grid_spacing = int(grid_spacing)
        assert ( 0 < grid_spacing < 360) and (360 % grid_spacing == 0), "grid_spacing should be a divisor of 360"
        self.grid_spacing = grid_spacing
        self.setup_grid()
        self.opt_queue = PriorityQueue()
        # try to use init_coords_M first, if not given, use M in engine's template
        # `for m in init_coords_M` doesn't work since m.measure_dihedrals will fail because it has different m.xyzs shape
        self.init_coords_M = [init_coords_M[i] for i in range(len(init_coords_M))] if init_coords_M != None else [self.engine.M]
        self.verbose = verbose
        # dictionary that stores the lowest energy for each grid point
        self.grid_energies = DefaultMaxDict()
        # dictionary that stores the geometries corresponding to lowest energy for each grid point
        self.grid_final_geometries = dict()
        # save current path as the rootpath
        self.rootpath = self.engine.rootpath = os.getcwd()


    #--------------------
    #  General methods
    #--------------------
    def get_dihedral_id(self, molecule, check_grid_id=None):
        """
        Compute the closest grid ID for molecule (only first frame)
        If check_grid_id is given, will perform a check if the computed dihedral_values are close to the grid_id provided
        """
        dihedral_values = np.array([molecule.measure_dihedrals(*d)[0] for d in self.dihedrals])
        if check_grid_id != None:
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
        gs = self.grid_spacing
        neighbor_gridids = []
        for i_dim in range(len(grid_id)):
            lower_neighbor = list(grid_id)
            lower_neighbor[i_dim] = normalize_dihedral(grid_id[i_dim] - gs)
            neighbor_gridids.append(tuple(lower_neighbor))
            higher_neighbor = list(grid_id)
            higher_neighbor[i_dim] = normalize_dihedral(grid_id[i_dim] + gs)
            neighbor_gridids.append(tuple(higher_neighbor))
        return tuple(neighbor_gridids)

    def grid_full_neighbors(self, grid_id):
        """ Take a center grid id, return all the neighboring grid ids, in all dimensions """
        gs = self.grid_spacing
        neighbor_gids_each_dim = []
        for gid_each_dim in grid_id:
            lower_neighbor = normalize_dihedral(gid_each_dim - gs)
            higher_neighbor = normalize_dihedral(gid_each_dim + gs)
            neighbor_gids_each_dim.append((lower_neighbor, higher_neighbor))
        return tuple(itertools.product(*neighbor_gids_each_dim))

    #----------------------
    # Initializing methods
    #----------------------

    def setup_grid(self):
        """
        Set up grid ids, each as a tuple with size corresponding to grid dimension. i.e.
        1-D: grid_ids = ( (-165, ), (-150, ), ... (180, )  )
        2-D: grid_ids = ( (-165,-165), (-165,-150), ... (180,180)  )
        """
        grid_dim = len(self.dihedrals)
        gs = self.grid_spacing
        grid_1D = range(-180+gs, 180+gs, gs)
        self.grid_ids = tuple(itertools.product(*[grid_1D]*grid_dim))


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
        # setup tmp folders
        self.create_tmp_folder()
        # dictionary that saves the grid_id for each running job
        self.running_job_path_id = dict()
        energy_decrease_thresh = 0.00001
        # print scan status interval
        start_time = last_print_time = time.time()
        show_scan_status_interval = 10
        while True:
            # Launch all jobs in self.opt_queue
            self.running_job_path_id.update(self.launch_opt_jobs())
            # wait until any job finishes, take out from self.running_job_path_id
            finished_job_path_ids = self.wait_extract_finished_jobs()
            # load molecule object from finished jobs
            best_grid_m = dict()
            for job_path, job_grid_id in finished_job_path_ids.items():
                m = self.engine.load_task_result_m(job_path)
                # we will check here if the optimized structure has the desired dihedral ids
                grid_id = self.get_dihedral_id(m, check_grid_id=job_grid_id)
                if m.qm_energy < self.grid_energies[grid_id] - energy_decrease_thresh:
                    if self.verbose:
                        print("Energy for grid_id %s dropped %f --> %f" % (str(grid_id), self.grid_energies[grid_id], m.qm_energy))
                    self.grid_energies[grid_id] = m.qm_energy
                    self.grid_final_geometries[grid_id] = np.array(m.xyzs[0])
                    best_grid_m[grid_id] = m
            # create new tasks based on the best_grid_m of this iteration
            for grid_id, m in best_grid_m.items():
                # every neighbor grid point will get one new task
                for neighbor_gid in self.grid_neighbors(grid_id):
                    task = m, neighbor_gid
                    # all jobs are pushed with the same priority for now, can be adjusted here
                    self.opt_queue.push(task)
            # check if it's time to show the status
            current_time = time.time()
            if self.verbose and current_time - last_print_time > show_scan_status_interval:
                print("Scan Status at %d s" % (current_time-start_time))
                print(self.draw_ascii_image())
                last_print_time = current_time
            # check if all jobs finished
            if len(self.opt_queue) == 0 and len(self.running_job_path_id) == 0:
                print("All optimizations converged at lowest energy. Job Finished!")
                self.finish()
                break


    #----------------------------------
    # Utility methods Called by Master
    #----------------------------------

    def push_initial_opt_tasks(self):
        """
        Push a set of initial tasks to self.opt_queue
        """
        for m in self.init_coords_M:
            grid_id = self.get_dihedral_id(m)
            task = (m, grid_id)
            self.opt_queue.push(task)
        if self.verbose:
            print("%d initial tasks pushed to opt_queue" % len(self.init_coords_M))

    def create_tmp_folder(self):
        assert hasattr(self, 'grid_ids'), 'Call self.setup_grid() first'
        os.chdir(self.rootpath)
        tmp_folder_name = 'opt_tmp'
        if os.path.isdir(tmp_folder_name):
            shutil.rmtree(tmp_folder_name)
        os.mkdir(tmp_folder_name)
        tmp_folder_dict = dict()
        for grid_id in self.grid_ids:
            tname = 'gid_' + '_'.join('%03d' % gid for gid in grid_id)
            tmp_folder_path = os.path.join(tmp_folder_name, tname)
            os.mkdir(tmp_folder_path)
            tmp_folder_dict[grid_id] = tmp_folder_path
        self.tmp_folder_dict = tmp_folder_dict

    def launch_opt_jobs(self):
        """
        Launch constrained optimizations for molecules in opt_queue
        The current opt_queue will be cleaned up
        Return a dictionary that contains path and grid_ids: { path0: grid_id0, path1: grid_id1 }
        """
        new_job_path_ids = dict()
        while len(self.opt_queue) > 0:
            m, grid_id = self.opt_queue.pop()
            job_path = self.launch_constrained_opt(m, grid_id)
            new_job_path_ids[job_path] = grid_id
        return new_job_path_ids

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
        self.engine.M = molecule
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
        next_job_id = len(existing_jobs) + 1
        job_path = os.path.join(tmp_path, str(next_job_id))
        os.mkdir(job_path)
        return job_path

    def wait_extract_finished_jobs(self):
        """
        Interface with engine to check if any job finished
        Will wait infinitely here until at least one job finished
        The finished job paths will be removed from self.running_job_path_id
        Return a dictionary of {finished job path: grid_id, ..}
        """
        if len(self.running_job_path_id) == 0:
            print("No jobs running, returning")
            return
        while True:
            finished_path_set = self.engine.find_finished_jobs(self.running_job_path_id)
            if len(finished_path_set) > 0:
                if self.verbose:
                    print("Find finished jobs:", finished_path_set)
                finished_job_path_ids = dict()
                for job_path in finished_path_set:
                    finished_job_path_ids[job_path] = self.running_job_path_id.pop(job_path)
                return finished_job_path_ids

    def finish(self):
        """ Write qdata.txt and scan.xyz file based on converged scan results """
        m = Molecule()
        m.elem = list(self.engine.M.elem)
        m.qm_energies, m.xyzs, m.comms = [], [], []
        for gid in self.grid_ids:
            m.qm_energies.append(self.grid_energies[gid])
            m.xyzs.append(self.grid_final_geometries[gid])
            m.comms.append("Dihedral %s Energy %f" % (str(gid), self.grid_energies[gid]))
        m.write('qdata.txt')
        print("Final scan energies are written to qdata.txt")
        m.write('scan.xyz')
        print("Final scan energies are written to scan.xyz")

    def draw_ascii_image(self):
        """ Return a string with ASCII colors showing current running status """
        if not hasattr(self, 'grid_energies') or not hasattr(self, 'running_job_path_id'):
            return ""
        gs = self.grid_spacing
        grid_1D = range(-180+gs, 180+gs, gs)
        len1d = len(grid_1D)
        grid_dim = len(self.dihedrals)
        result_str = ""
        count = 0
        running_job_ids = set(self.running_job_path_id.values())
        for grid_id in itertools.product(*[grid_1D]*grid_dim):
            symbol = ' -'
            if grid_id in running_job_ids:
                symbol = ' \033[0;33m+\033[0m' # orange for running jobs
            elif grid_id in self.grid_energies:
                symbol = ' \033[0;36mo\033[0m' # cyan for finished jobs
            result_str += symbol
            count += 1
            for i_dim in range(1, grid_dim):
                if count % (len1d)**i_dim == 0:
                    result_str += '\n'
        return result_str





    #----------------------------------
    # End of the DihedralScanner class
    #----------------------------------

class DefaultMaxDict(dict):
    def __missing__(self, key):
        return float("inf")


def normalize_dihedral(d):
    """ Normalize any number to the range (-180, 180], including 180 """
    return d + (180-d)//360*360

def load_dihedralfile(dihedralfile):
    """
    Load definition of dihedral from a text file, i.e. Loading the file

    # dihedral definition by atom indices starting from 0
    # i     j      k     l
      0     1      2     3
      1     2      3     4

    Will return dihedral_idxs = [(0,1,2,3), (1,2,3,4)]
    """
    dihedral_idxs = []
    with open(dihedralfile) as infile:
        for line in infile:
            line = line.strip()
            if line[0] == '#': continue
            dihedral_idxs.append([int(i) for i in line.split()])
    return dihedral_idxs

def create_engine(enginename, inputfile=None, work_queue_port=None, native_opt=False):
    """
    Function to create a QM Engine object with work_queue and geomeTRIC setup.
    This is intentionally left outside of DihedralScanner class, because multiple DihedralScanner could share the same engine
    """
    engine_dict = {'psi4': EnginePsi4, 'qchem': EngineQChem, 'terachem':EngineTerachem}
    # initialize a work_queue
    if work_queue_port != None:
        from WQtools import WorkQueue
        work_queue = WorkQueue(work_queue_port)
    else:
        work_queue = None
    engine = engine_dict[enginename](inputfile, work_queue, native_opt=native_opt)
    return engine

def main():
    import argparse, sys
    parser = argparse.ArgumentParser(description="Potential energy scan of dihedral angle from 1 to 360 degree", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile', type=str, help='Input template file for QMEngine. Geometry will be used as starting point for scanning.')
    parser.add_argument('dihedralfile', type=str, help='File defining all dihedral angles to be scanned.')
    parser.add_argument('--init_coords', type=str, help='File contain a list of geometries, that will be used as multiple starting points, overwriting the geometry in input file.')
    parser.add_argument('-g', '--grid_spacing', type=int, default=15, help='Grid spacing for dihedral scan, i.e. every 15 degrees')
    parser.add_argument('-e', '--engine', type=str, default="psi4", choices=['qchem', 'psi4', 'terachem'], help='Engine for running scan')
    parser.add_argument('--native_opt', action='store_true', default=False, help='Use QM program native constrained optimization algorithm. This will turn off geomeTRIC package.')
    parser.add_argument('--wq_port', type=int, default=None, help='Specify port number to use Work Queue to distribute optimization jobs.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print more information while running.')
    args = parser.parse_args()

    # print input command for reproducibility
    print(' '.join(sys.argv))

    # parse the dihedral file
    dihedral_idxs = load_dihedralfile(args.dihedralfile)

    # create QM Engine, and WorkQueue object if provided port
    engine = create_engine(args.engine, inputfile=args.inputfile, work_queue_port=args.wq_port, native_opt=args.native_opt)

    # load init_coords if provided
    init_coords_M = Molecule(args.init_coords) if args.init_coords else None

    # create DihedralScanner object
    scanner = DihedralScanner(engine, dihedrals=dihedral_idxs, grid_spacing=args.grid_spacing, init_coords_M=init_coords_M, verbose=args.verbose)
    # Run the scan!
    scanner.master()
    # After finish, print result
    print("Dihedral scan is finished!")
    print(" Grid ID                Energy")
    for grid_id in sorted(scanner.grid_energies.keys()):
        print("  %-20s %.10f" % (str(grid_id), scanner.grid_energies[grid_id]))

def test():
    engine = create_engine('psi4')
    for dim in range(1, 4):
        print("Testing %d-D scan setup" % dim)
        dihedrals = [list(range(d, d+4)) for d in range(dim)]
        scanner = DihedralScanner(engine, dihedrals=dihedrals, grid_spacing=90)
        gid = scanner.grid_ids[0]
        assert len(scanner.grid_ids) == 4**dim and len(gid) == dim
        assert len(scanner.grid_neighbors(gid)) == 2**dim
    print("All tests passed!")


if __name__ == "__main__":
    main()

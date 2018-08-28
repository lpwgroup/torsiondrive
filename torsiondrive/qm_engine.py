import os
import subprocess

import numpy as np
from geometric.molecule import Molecule


def check_all_float(iterable):
    try:
        [float(i) for i in iterable]
        return True
    except ValueError:
        return False


class QMEngine(object):
    def __init__(self, input_file=None, work_queue=None, native_opt=False):
        self.temp_type = None # will be set to either "gradient" or "optimize" later
        self.work_queue = work_queue
        self.native_opt = native_opt
        self.rootpath = os.getcwd()
        if input_file is not None:
            self.load_input(input_file)
        else:
            self.M = Molecule()

    def load_input(self, input_file):
        raise NotImplementedError

    def set_dihedral_constraints(self, dihedral_idx_values):
        """
        set arbitrary number of dihedrals to be constrained
        input:
        -------
        dihedral_idx_values: list of (id1, id2, id3, id4, value), where id are atom indices
        example: dihedral_idx_values = [(1,2,3,4,90), (2,3,4,5,100)]
        """
        assert isinstance(dihedral_idx_values, list) or isinstance(dihedral_idx_values, tuple)
        assert len(dihedral_idx_values) > 0
        assert len(dihedral_idx_values[0]) == 5
        self.dihedral_idx_values = dihedral_idx_values

    def write_constraints_txt(self):
        """ write a constraints.txt file for geomeTRIC """
        with open('constraints.txt', 'w') as outfile:
            outfile.write("$set\n")
            for d1, d2, d3, d4, v in self.dihedral_idx_values:
                # geomeTRIC use atomic index starting from 1
                outfile.write("dihedral %d %d %d %d %f\n" % (d1+1, d2+1, d3+1, d4+1, v))

    def load_geomeTRIC_output(self):
        """ Load the optimized geometry and energy into a new molecule object and return """
        m = Molecule('opt.xyz')
        with open('energy.txt') as infile:
            m.qm_energies = [float(infile.read())]
        return m

    def launch_optimize(self, job_path=None):
        """ launch an optimization job inside job_path """
        orig_dir = os.getcwd()
        if job_path is not None:
            os.chdir(job_path)
        if self.native_opt:
            self.optimize_native()
        else:
            self.optimize_geomeTRIC()
        os.chdir(orig_dir)

    def load_task_result_m(self, job_path=None):
        """
        Load the result of optimization into a Molecule object, from job_path
        Return the Molecule object
        """
        orig_dir = os.getcwd()
        if job_path is not None:
            os.chdir(job_path)
        if self.native_opt:
            m = self.load_native_output()
        else:
            m = self.load_geomeTRIC_output()
        os.chdir(orig_dir)
        return m

    def run(self, cmd, input_files=None, output_files=None):
        """ Execute a command locally or remotely, based on whether self.work_queue is set """
        if input_files is None:
            input_files = []
        if output_files is None:
            output_files = []
        if self.work_queue is None:
            subprocess.check_call(cmd, shell=True)
        else:
            self.work_queue.submit(cmd, input_files, output_files)

    def find_finished_jobs(self, running_job_path_id, wait_time=10):
        """ Find finished jobs in job_path_set, return a set of job paths """
        assert wait_time > 0
        finished_path_set = set()
        if self.work_queue is None:
            # if running locally, all jobs must have finished already
            for path in running_job_path_id:
                finished_path_set.add(path)
        else:
            # if using work queue, take some time to collect finished job paths
            for t in range(wait_time):
                # check every second if a task finished in work_queue
                taskpath = self.work_queue.check_finished_task_path(wait_time=1)
                if taskpath is not None:
                    taskpath = os.path.relpath(taskpath, self.rootpath)
                    assert taskpath in running_job_path_id, "Finished task path should be one of the running job paths"
                    finished_path_set.add(taskpath)
        return finished_path_set

    # An empty engine will not run any real calculations
    # These functions will be defined in the subclasses
    def optimize_native(self):
        return

    def optimize_geomeTRIC(self):
        return

    def load_native_output(self):
        return


class EnginePsi4(QMEngine):
    def load_input(self, input_file):
        """ Load a Psi4 input file as a Molecule object into self.M
        Only xyz input coordinates are supported for now.
        Exmaple input file:

        memory 12 gb
        molecule {
        0 1
        H  -0.90095  -0.50851  -0.76734
        O  -0.72805   0.02496   0.02398
        O   0.72762   0.03316  -0.02696
        H   0.90782  -0.41394   0.81465
        units angstrom
        no_reorient
        symmetry c1
        }
        set globals {
            basis         6-31+g*
            freeze_core   True
            guess         sad
            scf_type      df
            print         1
        }
        set_num_threads(1)
        gradient('mp2')
        """
        coords = []
        elems = []
        reading_molecule, found_geo = False, False
        psi4_temp = [] # store a template of the input file for generating new ones
        with open(input_file) as psi4in:
            for line in psi4in:
                line_sl = line.strip().lower()
                if line_sl.startswith("molecule"):
                    reading_molecule = True
                    psi4_temp.append(line)
                elif reading_molecule is True:
                    ls = line.split()
                    if len(ls) == 4 and check_all_float(ls[1:]):
                        if not found_geo:
                            found_geo = True
                            psi4_temp.append("$!geometry@here")
                        # parse the xyz format
                        elems.append(ls[0])
                        coords.append(ls[1:4])
                    else:
                        psi4_temp.append(line)
                        if '}' in line:
                            reading_molecule = False
                            psi4_temp.append("$!optking@here")
                else:
                    psi4_temp.append(line)
                if  line_sl.startswith('gradient('):
                    self.temp_type = "gradient"
                elif line_sl.startswith('optimize('):
                    self.temp_type = "optimize"
        assert found_geo, "XYZ geometry not found in molecule block of %s" % input_file
        if self.native_opt:
            assert self.temp_type == 'optimize', "input_file should contain optimize() command to use native opt"
        else:
            assert self.temp_type == 'gradient', "input_file should contain gradient() command to use geomeTRIC"
        # self.psi4_temp will enable writing input files with new geometries
        self.psi4_temp = psi4_temp
        # here self.M can be and will be overwritten by external functions
        self.M = Molecule()
        self.M.elem = elems
        self.M.xyzs = [np.array(coords, dtype=float)]
        self.M.build_topology()

    def write_input(self, filename='input.dat'):
        """ Write output based on self.psi4_temp and self.M, using only geometry of the first frame """
        assert hasattr(self, 'psi4_temp'), "psi4_temp not found, call self.load_input() first"
        with open(filename, 'w') as outfile:
            for line in self.psi4_temp:
                if line == '$!geometry@here':
                    for e, c in zip(self.M.elem, self.M.xyzs[0]):
                        outfile.write("%-7s %13.7f %13.7f %13.7f\n" % (e, c[0], c[1], c[2]))
                elif line == '$!optking@here':
                    if hasattr(self, 'optkingStr'):
                        outfile.write(self.optkingStr)
                else:
                    outfile.write(line)

    def optimize_native(self):
        """ run the constrained optimization using native Optking, in 2 steps:
        1. write a optimization job input file.
        2. run the job
        """
        assert self.temp_type == 'optimize', "To use native optimization, the input file should have optimize() in it"
        # add the optking command
        self.optkingStr = '\nset optking {\n  fixed_dihedral = ("\n'
        for d1, d2, d3, d4, v in self.dihedral_idx_values:
            # Optking use atom index starting from 1
            self.optkingStr += '        %d  %d  %d  %d  %f\n' % (d1+1, d2+1, d3+1, d4+1, v)
        self.optkingStr += '  ")\n}\n'
        # write input file
        self.write_input('input.dat')
        # run the job
        self.run('psi4 input.dat -o output.dat', input_files=['input.dat'], output_files=['output.dat'])

    def optimize_geomeTRIC(self):
        """ run the constrained optimization using geomeTRIC package, in 3 steps:
        1. Write a constraints.txt file.
        2. Write a gradient job input file.
        3. Run the job
        """
        assert self.temp_type == 'gradient', "To use geomeTRIC package, the input file should have gradient() in it"
        # step 1
        self.write_constraints_txt()
        # step 2
        self.write_input('input.dat')
        # step 3
        self.run('geometric-optimize --qccnv --reset --epsilon 0.0 --psi4 input.dat constraints.txt > optimize.log', input_files=['input.dat', 'constraints.txt'], output_files=['optimize.log', 'opt.xyz', 'energy.txt'])

    def load_native_output(self, filename='output.dat'):
        """ Load the optimized geometry and energy into a new molecule object and return """
        found_opt_result = False
        found_final_geo = False
        final_energy, elems, coords = None, [], []
        with open(filename) as outfile:
            for line in outfile:
                line = line.strip()
                if line.startswith('Final energy is'):
                    final_energy = float(line.split()[-1])
                elif line.startswith('Final optimized geometry and variables'):
                    found_opt_result = True
                elif found_opt_result:
                    ls = line.split()
                    if len(ls) == 4 and check_all_float(ls[1:]):
                        elems.append(ls[0])
                        coords.append(ls[1:4])
        if final_energy is None:
            raise RuntimeError("Final energy not found in %s" % filename)
        if len(elems) == 0 or len(coords) == 0:
            raise RuntimeError("Final geometry not found in %s" % filename)
        m = Molecule()
        m.elem = elems
        m.xyzs = [np.array(coords, dtype=float)]
        m.qm_energies = [final_energy]
        m.build_topology()
        return m


class EngineQChem(QMEngine):
    def load_input(self, input_file):
        """
        Load QChem input
        Example input file:

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
        """
        elems,coords = [], []
        reading_molecule, found_geo = False, False
        qchem_temp = [] # store a template of the input file for generating new ones
        with open(input_file) as qchemin:
            for line in qchemin:
                line_sl = line.strip().lower()
                if line_sl.startswith("$molecule"):
                    reading_molecule = True
                    qchem_temp.append(line)
                elif reading_molecule:
                    ls = line.split()
                    if len(ls) == 4 and check_all_float(ls[1:]):
                        if not found_geo:
                            found_geo = True
                            qchem_temp.append("$!geometry@here")
                        elems.append(ls[0])
                        coords.append(ls[1:])
                    else:
                        qchem_temp.append(line)
                    if line_sl.startswith('$end'):
                        reading_molecule = False
                        qchem_temp.append("$!optblock@here")
                else:
                    qchem_temp.append(line)
                if line_sl.startswith('jobtype'):
                    jobtype = line_sl.split()[1]
                    if jobtype.startswith('opt'):
                        self.temp_type = 'optimize'
                    elif jobtype.startswith('force'):
                        self.temp_type = 'gradient'
        if self.native_opt:
            assert self.temp_type == 'optimize', "input_file should be a opt job to use native opt"
        else:
            assert self.temp_type == 'gradient', "input_file should be a gradient job to use geomeTRIC"
        # self.qchem_temp will enable writing input files with new geometries
        self.qchem_temp = qchem_temp
        # here self.M can be and will be overwritten by external functions
        self.M = Molecule()
        self.M.elem = elems
        self.M.xyzs = [np.array(coords, dtype=float)]
        self.M.build_topology()

    def write_input(self, filename='qc.in'):
        """ Write QChem input using Molecule Class """
        assert hasattr(self, 'qchem_temp'), "self.qchem_temp not set, call load_input() first"
        with open(filename, 'w') as outfile:
            for line in self.qchem_temp:
                if line == '$!geometry@here':
                    for e, c in zip(self.M.elem, self.M.xyzs[0]):
                        outfile.write("%-7s %13.7f %13.7f %13.7f\n" % (e, c[0], c[1], c[2]))
                elif line == "$!optblock@here":
                    if hasattr(self, 'optblockStr'):
                        # self.optblockStr will be set by self.optimize_native()
                        outfile.write(self.optblockStr)
                else:
                    outfile.write(line)

    def optimize_native(self):
        """
        Run the constrained optimization, following QChem 5.0 manual.
        1. write a optimization job input file.
        2. run the job
        """
        assert self.temp_type == 'optimize', "To use native optimization, the input file be an opt job"
        # add the $opt block
        self.optblockStr = '\n$opt\nCONSTRAINT\n'
        for d1, d2, d3, d4, v in self.dihedral_idx_values:
            # Optking use atom index starting from 1
            self.optblockStr += 'tors  %d  %d  %d  %d  %f\n' % (d1+1, d2+1, d3+1, d4+1, v)
        self.optblockStr += 'ENDCONSTRAINT\n$end\n'
        # write input file
        self.write_input('qc.in')
        # run the job
        self.run('qchem qc.in qc.out > qc.log', input_files=['qc.in'], output_files=['qc.out', 'qc.log'])

    def optimize_geomeTRIC(self):
        """ run the constrained optimization using geomeTRIC package, in 3 steps:
        1. Write a constraints.txt file.
        2. Write a gradient job input file.
        3. Run the job
        """
        assert self.temp_type == 'gradient', "To use geomeTRIC package, the input file should have gradient() in it"
        # step 1
        self.write_constraints_txt()
        # step 2
        self.write_input('qc.in')
        # step 3
        self.run('geometric-optimize --qccnv --reset --epsilon 0.0 --qchem qc.in constraints.txt > optimize.log', input_files=['qc.in', 'constraints.txt'], output_files=['optimize.log', 'opt.xyz', 'energy.txt'])

    def load_native_output(self, filename='qc.out'):
        """ Load the optimized geometry and energy into a new molecule object and return """
        m = Molecule(filename, ftype="qcout")[-1]
        return m


class EngineTerachem(QMEngine):
    def load_input(self, input_file):
        """
        Load TeraChem input
        Example input file:

        coordinates start.xyz
        run gradient
        basis 6-31g*
        method rb3lyp
        charge 0
        spinmult 1
        dispersion yes
        scf diis+a
        maxit 50
        """
        self.tera_temp = []
        geo_file = None
        with open(input_file) as terain:
            for line in terain:
                # we don't need to change the temp
                self.tera_temp.append(line)
                linest = line.strip()
                if not linest: continue
                key, value = linest.lower().split(None, 1)
                if key == 'coordinates':
                    geo_file = value
                elif key == 'run':
                    if value == 'gradient':
                        self.temp_type = 'gradient'
                    elif value == 'minimize':
                        self.temp_type = 'optimize'
        # place holder for writing native constraints
        self.tera_temp.append('$!constraints@here')
        # check input
        assert geo_file, 'coordinates key not found in input file %s' % input_file
        if self.native_opt:
            assert self.temp_type == 'optimize', "input_file should be a opt job to use native opt"
        else:
            assert self.temp_type == 'gradient', "input_file should be a gradient job to use geomeTRIC"
        # load molecule from separate file, one frame only
        self.M = Molecule(geo_file)[0]
        # store the name of geo_file
        self.tera_geo_file = geo_file

    def write_input(self):
        """ Write TeraChem input files, i.e. run.in and start.xyz """
        assert hasattr(self, 'tera_temp'), "self.tera_temp not set, call load_input() first"
        assert hasattr(self, 'tera_geo_file'), "self.tera_temp not set, call load_input() first"
        with open('run.in', 'w') as terain:
            for line in self.tera_temp:
                if line == "$!constraints@here":
                    if hasattr(self, 'constraintsStr'):
                        # self.optblockStr will be set by self.optimize_native()
                        terain.write(self.constraintsStr)
                else:
                    terain.write(line)
        self.M.write(self.tera_geo_file)

    def optimize_native(self):
        """
        Run the constrained optimization, following QChem 5.0 manual.
        1. write a optimization job input file.
        2. run the job
        """
        assert self.temp_type == 'optimize', "To use native optimization, the input file be an opt job"
        # add the $opt block
        self.constraintsStr = '\n$constraint_set\n'
        for d1, d2, d3, d4, v in self.dihedral_idx_values:
            # Optking use atom index starting from 1
            self.constraintsStr += 'dihedral %f %d_%d_%d_%d\n' % (v, d1+1, d2+1, d3+1, d4+1)
        self.constraintsStr += '$end\n'
        # write input file
        self.write_input()
        # run the job
        self.run('terachem run.in > run.out', input_files=['run.in', self.tera_geo_file], output_files=['run.out', 'scr'])

    def optimize_geomeTRIC(self):
        """ run the constrained optimization using geomeTRIC package, in 3 steps:
        1. Write a constraints.txt file.
        2. Write a gradient job input file.
        3. Run the job
        """
        assert self.temp_type == 'gradient', "To use geomeTRIC package, the input file should have gradient() in it"
        # step 1
        self.write_constraints_txt()
        # step 2
        self.write_input()
        # step 3
        self.run('geometric-optimize --qccnv --reset --epsilon 0.0 run.in constraints.txt > optimize.log', input_files=['run.in', self.tera_geo_file, 'constraints.txt'], output_files=['optimize.log', 'opt.xyz', 'energy.txt'])

    def load_native_output(self):
        """ Load the optimized geometry and energy into a new molecule object and return """
        m = Molecule('scr/optim.xyz')[-1]
        # read the energy from optim.xyz comment line
        m.qm_energies = [float(m.comms[0].split(None, 1)[0])]
        return m

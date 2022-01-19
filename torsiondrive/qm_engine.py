import os
import shutil
import subprocess
import numpy as np
import copy
from geometric.molecule import Molecule
from torsiondrive.extra_constraints import build_geometric_constraint_string, build_terachem_constraint_string

def check_all_float(iterable):
    try:
        [float(i) for i in iterable]
        return True
    except ValueError:
        return False

class QMEngine(object):
    def __init__(self, input_file=None, work_queue=None, native_opt=False, extra_constraints=None):
        self.temp_type = None # will be set to either "gradient" or "optimize" later
        self.work_queue = work_queue
        self.native_opt = native_opt
        self.extra_constraints = extra_constraints
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
        if self.extra_constraints is None:
            constraints_string = "$set\n"
            for d1, d2, d3, d4, v in self.dihedral_idx_values:
                # geomeTRIC use atomic index starting from 1
                constraints_string += f"dihedral {d1+1} {d2+1} {d3+1} {d4+1} {float(v)}\n"
        else:
            constraints_string = build_geometric_constraint_string(self.extra_constraints, self.dihedral_idx_values)
        with open('constraints.txt', 'w') as outfile:
            outfile.write(constraints_string)

    def load_geomeTRIC_output(self):
        """ Load the optimized geometry and energy into a new molecule object and return """
        # the name of the file is consistent with the --prefix tdrive option,
        # this also requires the input file NOT be named to sth like tdrive.in
        # otherwise the output will become tdrive_optim.xyz
        if not os.path.isfile('qdata.txt'):
            raise OSError("geomeTRIC output qdata.txt file not found")
        m = Molecule('qdata.txt')[-1]
        # copy the m.elem since qdata.txt does not have it
        m.elem = self.M.elem
        # check the data loaded
        assert len(m.qm_energies) == 1
        assert len(m.qm_grads) == 1 and m.qm_grads[0].shape == self.M.xyzs[0].shape
        m.build_topology()
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
            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError:
                print('Run failed.')
                # Create a Failed file to indicate a failed calculation.
                open('Failed', 'a').close()
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

    # These functions should be defined in the subclasses
    def optimize_native(self):
        raise NotImplementedError

    def optimize_geomeTRIC(self):
        raise NotImplementedError

    def load_native_output(self):
        raise NotImplementedError

class EngineBlank(QMEngine):
    """ A blank engine only used in testing """
    def optimize_native(self):
        return

    def optimize_geomeTRIC(self):
        return

    def load_native_output(self):
        return Molecule()

class EngineOpenMM(QMEngine):
    def load_input(self, input_file):
        """Input file is the name of the pdb file with the coords in we also require that the xml has the same name"""

        self.m_pdb = Molecule(input_file)[0]
        self.M = copy.deepcopy(self.m_pdb)

        xml_name = os.path.splitext(input_file)[0] + '.xml'
        # Check the xml file is present
        assert os.path.exists(xml_name) is True, "OpenMM requires a pdb and xml file, ensure you have both in the current folder with the same prefix"
        with open(xml_name) as f:
            self.xml_content = f.read()

    def write_input(self):
        """Write a pdb file with the latest geometry and the input xml file"""

        self.m_pdb.xyzs[0] = self.M.xyzs[0]
        self.m_pdb.write('input.pdb')
        with open('input.xml', 'w') as out:
                out.write(self.xml_content)

    def optimize_geomeTRIC(self):
        """ run the constrained optimization using geomeTRIC package, in 3 steps:
        1. Write a constraints.txt file.
        2. Write a gradient job input file.
        3. Run the job
        """
        # sep 1
        self.write_constraints_txt()
        # step 2
        self.write_input()
        # set3
        self.run('geometric-optimize --prefix tdrive --qccnv --reset --epsilon 0.0 --enforce 0.1 --qdata --pdb '
                 'input.pdb --openmm input.xml constraints.txt',
                 input_files=['input.xml', 'input.pdb', 'constraints.txt'],
                 output_files=['tdrive.log', 'tdrive.xyz', 'qdata.txt'])

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
        if self.extra_constraints is not None:
            raise RuntimeError('extra constraints not supported in Psi4 native optimizations')
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
        cmd = 'geometric-optimize --prefix tdrive --qccnv --reset --epsilon 0.0 --enforce 0.1 --qdata --psi4 input.dat constraints.txt'
        self.run(cmd, input_files=['input.dat', 'constraints.txt'], output_files=['tdrive.log', 'tdrive.xyz', 'qdata.txt'])

    def load_native_output(self, filename='output.dat'):
        """ Load the optimized geometry and energy into a new molecule object and return """
        found_opt_result = False
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

class EngineGaussian(QMEngine):
    def __init__(self, input_file=None, work_queue=None, native_opt=False, extra_constraints=None, exe=None):
        super().__init__(input_file, work_queue, native_opt, extra_constraints)
        # Check which version of gaussain we have access to
        if exe.lower() in ("g09", "g16"):
            self.gaussian_exe = exe.lower()
        else:
            raise ValueError("Only g16 and g09 are supported.")

    def load_input(self, input_file):
        """
        !!!only Cartesian molecule specification is supported at the moment!!!
        Load Gaussian09 input file, note blank lines at the bottom of the file are required
        Example input file:

        %Mem=6GB
        %NProcShared=2
        %Chk=lig
        # B3LYP/6-31G(d) Opt=ModRedundant

        water energy

        0   1
        O  -0.464   0.177   0.0
        H  -0.464   1.137   0.0
        H   0.441  -0.143   0.0


        """
        reading_molecule, found_geo = False, False
        gauss_temp = []  # store a template of the input file for generating new ones
        with open(input_file) as gauss_in:
            for line in gauss_in:
                ls = line.split()
                if len(ls) == 4 and check_all_float(ls[1:]):
                    reading_molecule = True
                    if not found_geo:
                        found_geo = True
                        gauss_temp.append("$!geometry@here")

                elif reading_molecule:
                    if line.strip() == '':
                        reading_molecule = False
                        gauss_temp.append(line)
                        gauss_temp.append("$!optblock@here")

                elif "%chk" in line.lower():
                    # we need to overwrite the input to make the name consistent
                    gauss_temp.append("%Chk=ligand\n")

                else:
                    gauss_temp.append(line)

                if 'opt=modredundant' in line.lower():
                    self.temp_type = 'optimize'
                elif "force=nostep" in line.lower():
                    self.temp_type = "gradient"
        assert found_geo, "XYZ geometry not found in molecule block of %s" % input_file
        if self.native_opt:
            assert self.temp_type == 'optimize', "input_file should be a opt job to use native opt include the Opt=ModRedundant flag"
        # make sure the checkpoint file name is included
        if not any("%chk" in command.lower() for command in gauss_temp):
            gauss_temp.insert(0, "%Chk=ligand\n")

        self.gauss_temp = gauss_temp
        self.M = Molecule(input_file)

    def optimize_geomeTRIC(self):
        """ run the constrained optimization using geomeTRIC package, in 3 steps:
        1. Write a constraints.txt file.
        2. Write a gradient job input file.
        3. Run the job
        """
        assert self.temp_type == 'gradient', "To use geomeTRIC package, the input file should have Force=NoStep in it"
        # step 1
        self.write_constraints_txt()
        # step 2
        self.write_input('input.com')
        # step 3
        cmd = 'geometric-optimize --prefix tdrive --qccnv yes --reset yes --epsilon 0.0 --enforce 0.1 --qdata yes --engine gaussian input.com constraints.txt'
        self.run(cmd, input_files=['input.com', 'constraints.txt'],
                 output_files=['tdrive.log', 'tdrive.xyz', 'qdata.txt'])

    def write_input(self, filename='gaussian.com'):
        """ Write Gaussian input using Molecule Class """
        assert hasattr(self, 'gauss_temp'), "self.gauss_temp not set, call load_input() first"
        with open(filename, 'w') as outfile:
            for line in self.gauss_temp:
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
        Run the constrained optimization, following Gaussian09 manual.
        1. write a optimization job input file.
        2. run the job
        """
        assert self.temp_type == 'optimize', "To use native optimization, the input file must be an opt job"
        assert hasattr(self, 'gaussian_exe'), 'The version of gaussian could not be determined!'
        if self.extra_constraints is not None:
            raise RuntimeError('extra constraints not supported in Gaussian native optimizations')
        self.optblockStr =''
        for d1, d2, d3, d4, v in self.dihedral_idx_values:
            self.optblockStr += f'{d1 + 1} {d2 + 1} {d3 + 1} {d4 + 1} ={v:.3f} B\n'  # Build the angle
            self.optblockStr += f'{d1 + 1} {d2 + 1} {d3 + 1} {d4 + 1} F\n\n'           # Freeze the angle
        # write input file
        self.write_input('gaussian.com')
        # run the job
        self.run(f'{self.gaussian_exe} < gaussian.com > gaussian.log && formchk ligand.chk ligand.fchk', input_files=['gaussian.com'],
                 output_files=['gaussian.log', 'ligand.fchk'])

    def load_native_output(self, filename='ligand.fchk', filename2='gaussian.log'):
        """ Load the optimized geometry and energy into a new molecule object and return """
        # Check the log file to see if the optimization was successful
        opt_result = False
        final_energy, elems, coords = None, [], []
        with open(filename2) as logfile:
            for line in logfile:
                if 'Optimization completed' in line:
                    opt_result = True
                    break

        if not opt_result:
            raise RuntimeError("Geometry optimization failed in %s" % filename2)

        # Now we want to get the optimized structure from the fchk file as this is more reliable
        end_xyz_pos = None
        with open(filename) as outfile:
            for i, line in enumerate(outfile):
                if 'Current cartesian coordinates' in line:
                    num_xyz = int(line.split()[5])
                    end_xyz_pos = int(np.ceil(num_xyz/5)+i+1)
                elif end_xyz_pos is not None and i < end_xyz_pos:
                    coords.extend([float(num) * 0.529177 for num in line.strip('\n').split()])
                elif 'Total Energy' in line:
                    final_energy = float(line.split()[3])

        if end_xyz_pos is None:
            raise RuntimeError('Cannot locate coordinates in ligand.fchk file.')

        # Make sure we have all of the coordinates
        assert len(coords) == num_xyz, "Could not extract the optimised geometry"

        if final_energy is None:
            raise RuntimeError("Final energy not found in %s" % filename)

        m = Molecule()
        m.elem = self.M.elem
        m.xyzs = [np.reshape(coords, (int(len(m.elem)), 3))]
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
        if self.extra_constraints is not None:
            raise RuntimeError('extra constraints not supported in Q-Chem native optimizations')
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
        cmd = 'geometric-optimize --prefix tdrive --qccnv --reset --epsilon 0.0 --enforce 0.1 --qdata --qchem qc.in constraints.txt'
        self.run(cmd, input_files=['qc.in', 'constraints.txt'], output_files=['tdrive.log', 'tdrive.xyz', 'qdata.txt'])

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
                        # self.constraintsStr will be set by self.optimize_native()
                        terain.write(self.constraintsStr)
                else:
                    terain.write(line)
        self.M.write(self.tera_geo_file)

    def optimize_native(self):
        """
        Run the constrained optimization.
        1. write a optimization job input file.
        2. run the job
        """
        assert self.temp_type == 'optimize', "To use native optimization, the input file be an opt job"
        if self.extra_constraints is None:
            self.constraintsStr = '\n$constraint_set\n'
            for d1, d2, d3, d4, v in self.dihedral_idx_values:
                # TeraChem use atom index starting from 1
                self.constraintsStr += f"dihedral {float(v)} {d1+1}_{d2+1}_{d3+1}_{d4+1}\n"
            self.constraintsStr += '$end\n'
        else:
            self.constraintsStr = build_terachem_constraint_string(self.extra_constraints, self.dihedral_idx_values)
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
        cmd = 'geometric-optimize --prefix tdrive --qccnv --reset --epsilon 0.0 --enforce 0.1 --qdata run.in constraints.txt'
        self.run(cmd, input_files=['run.in', self.tera_geo_file, 'constraints.txt'], output_files=['tdrive.log', 'tdrive.xyz', 'qdata.txt'])

    def load_native_output(self):
        """ Load the optimized geometry and energy into a new molecule object and return """
        m = Molecule('scr/optim.xyz')[-1]
        # read the energy from optim.xyz comment line
        m.qm_energies = [float(m.comms[0].split(maxsplit=1)[0])]
        return m

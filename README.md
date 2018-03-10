# DihedralScanner
Dihedral Scanner Class Developed From https://github.com/lpwgroup/crank

DihedralScanner class is designed to create a dihedral grid, and fill in optimized geometries and energies into the grid, by running wavefront propagations of constrained optimizations.

## Dependencies

* Numpy

* [geomeTRIC](https://github.com/leeping/geomeTRIC) molecule.py, also to enable the `geometric-optimize` command

* [cctools.work_queue](https://github.com/cooperative-computing-lab/cctools) [Optional] (to enable the distributed computing feature)

## Install
No need to install.

## Run
`python DihedralScanner.py <InputFile> <Dihedrals.txt> --init_coords <init_coords.xyz> -g <grid_spacing> -e <Engine> (--native_opt) (--wq_port XXX) (-v) > scan.log`

### Input Parameters:

#### positional arguments:

`<InputFile>`: An Psi4, QChem or Terachem input file, servering as a template for constrained optimizations.

`<Dihedrals.txt>`: A txt file containing the definition of dihedral angles, each as 4 atom indices in one line. The number of dihedrals determines the dimension of the scanned grid.

#### optional arguments:

`--init_coords <init_coords.xyz>`: File contains a trajectory of initial coordinates. Will be mapped to the closest grid point at beginning, replacing the geometry in `<InputFile>`.

`-g <grid_spacing>`: Integer number, divisor of 360, angle in degree between each grid point.

`-e <Engine>`: One of `psi4`, `qchem`, `terachem`

`--native_opt`: Flag, use quantum software's internal optimizer instead of geomeTRIC optimizer.

`--wq_port XXX`: Flag, use `cctools.work_queue` tool to distribute the calculations.

`-v`: Flag, turn on verbose printing, including the colorful status map :)

### Advanced Usage

The `DihedralScanner` Class works together with the `QMEngine` Class, both can be imported and incorporated into other python scripts, to enable more features, and automated scanning for multiple molecules.

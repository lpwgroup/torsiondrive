torsiondrive
==============================
[![Travis Build Status](https://travis-ci.org/lpwgroup/torsiondrive.png)](https://travis-ci.org/lpwgroup/torsiondrive)
[![codecov](https://codecov.io/gh/lpwgroup/torsiondrive/branch/master/graph/badge.svg)](https://codecov.io/gh/lpwgroup/torsiondrive/branch/master)

Dihedral scanner with wavefront propagation

## Dependencies

* Numpy

* [geomeTRIC](https://github.com/leeping/geomeTRIC) molecule.py, also to enable the `geometric-optimize` command

* [cctools.work_queue](https://github.com/cooperative-computing-lab/cctools) [Optional] (to enable the distributed computing feature)

## Install
`python setup.py install`

## Run
`torsiondrive-launch <InputFile> <Dihedrals.txt> --init_coords <init_coords.xyz> -g <grid_spacing> -e <Engine> (--native_opt) (--wq_port XXX) (-v) > scan.log`

## API call
`torsiondrive-api current_state.json`

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

#### Acknowledgements

Project based on the
[Computational Chemistry Python Cookiecutter](https://github.com/choderalab/cookiecutter-python-comp-chem)

#### Funding Information

The development of this code has been supported in part by the following grants and awards:

NIH Grant R01 AI130684-02

ACS-PRF 58158-DNI6

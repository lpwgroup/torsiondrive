# DihedralScanner
Dihedral Scanner Class Developed From https://github.com/lpwgroup/crank

DihedralScanner class is designed to create a dihedral grid, and fill in optimized geometries and energies into the grid, by running wavefront propagations of constrained optimizations.

## Dependencies

* Numpy

* [ForceBalance](https://github.com/leeping/forcebalance) (in particular, the molecule.py)

* [geomeTRIC](https://github.com/leeping/geomeTRIC)  to enable the `geometric-optimize` command

* [cctools.work_queue](https://github.com/cooperative-computing-lab/cctools) [Optional] (to enable the distributed computing feature)

## Install
No need to install.

## Run
`python DihedralScanner.py <InputFile> <Dihedrals.txt> -e <Engine> (--native_opt) (--wq_port XXX) (-v) > scan.log`

### Explanation (also shown in help menu `python DihedralScanner.py -h`) :

`<InputFile>`: An Psi4, QChem or Terachem input file, servering as a template for constrained optimizations.

`<Dihedrals.txt>`: A txt file containing the definition of dihedral angles, each as 4 atom indices in one line. The number of dihedrals determines the dimension of the scanned grid.

`<Engine>`: One of `psi4`, `qchem`, `terachem`

`(--native_opt)`: Optional flag, use quantum software's internal optimizer instead of geomeTRIC optimizer.

`(--wq_port XXX)`: Optional flag, use `cctools.work_queue` tool to distribute the calculations.

`(-v)`: Optional flag, turn on verbose printing, including the colorful status map :)

### Advance Usage

The `DihedralScanner` Class works together with the `QMEngine` Class, both can be imported and incorporated into other python scripts, to enable more features, and automated scanning for multiple molecules.

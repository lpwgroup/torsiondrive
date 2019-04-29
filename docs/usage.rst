Run TorsionDrive
======================


Using the Command Line
----------------------

Once installed, you can start torsiondrive scans from command line:

.. code-block:: console

    $ torsiondrive-launch -h
    usage: torsiondrive-launch [-h] [--init_coords INIT_COORDS]
                            [-g [GRID_SPACING [GRID_SPACING ...]]]
                            [-e {qchem,psi4,terachem}] [-c CONSTRAINTS]
                            [--native_opt] [--energy_thresh ENERGY_THRESH]
                            [--energy_upper_limit ENERGY_UPPER_LIMIT]
                            [--wq_port WQ_PORT] [--zero_based_numbering] [-v]
                            inputfile dihedralfile

    Potential energy scan of dihedral angle from 1 to 360 degree

    positional arguments:
    inputfile             Input template file for QMEngine. Geometry will be
                            used as starting point for scanning.
    dihedralfile          File defining all dihedral angles to be scanned.

    optional arguments:
    -h, --help            show this help message and exit
    --init_coords INIT_COORDS
                            File contain a list of geometries, that will be used
                            as multiple starting points, overwriting the geometry
                            in input file. (default: None)
    -g [GRID_SPACING [GRID_SPACING ...]], --grid_spacing [GRID_SPACING [GRID_SPACING ...]]
                            Grid spacing for dihedral scan, i.e. every 15 degrees,
                            multiple values will be mapped to each dihedral angle
                            (default: [15])
    -e {qchem,psi4,terachem}, --engine {qchem,psi4,terachem}
                            Engine for running scan (default: psi4)
    -c CONSTRAINTS, --constraints CONSTRAINTS
                            Provide a constraints file in geomeTRIC format for
                            additional freeze or set constraints (geomeTRIC or
                            TeraChem only) (default: None)
    --native_opt          Use QM program native constrained optimization
                            algorithm. This will turn off geomeTRIC package.
                            (default: False)
    --energy_thresh ENERGY_THRESH
                            Only activate grid points if the new optimization is
                            <thre> lower than the previous lowest energy (in
                            a.u.). (default: 1e-05)
    --energy_upper_limit ENERGY_UPPER_LIMIT
                            Only activate grid points if the new optimization is
                            less than <thre> higher than the global lowest energy
                            (in a.u.). (default: None)
    --wq_port WQ_PORT     Specify port number to use Work Queue to distribute
                            optimization jobs. (default: None)
    --zero_based_numbering
                            Use zero_based_numbering in dihedrals file. (default:
                            False)
    -v, --verbose         Print more information while running. (default: False)


Using the API (advanced)
------------------------

An API interface of torsiondrive is provided for interfacing with QCFractal servers.
The main difference of the API method is that the API is designed as a "service",
which generates one iteration of constrained optimizations each time.

.. code-block:: console

    $ torsiondrive-api -h
    usage: torsiondrive-api [-h] [-v] statefile

    Take a scan state and return the next set of optimizations

    positional arguments:
    statefile      File contains the current state in JSON format

    optional arguments:
    -h, --help     show this help message and exit
    -v, --verbose  Print more information while running. (default: False)

A json file containing the scan options and the "current state" of torsion scan is passed to the API,
then the API program will reproduce the **entire** torsion scan from the beginning, until some new optimiations are needed.

The new optimiations will be returned also in JSON format. If the scan is finished, the return will be empty.
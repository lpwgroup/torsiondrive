TorsionDrive Examples
=====================

Example runs of torsiondrive can be found in repository https://github.com/lpwgroup/torsiondrive_examples

1-D Examples
------------
Example input, output and running commands can be found in ``torsiondrive_examples/examples/hooh-1d``, including all combinations of
 - Quantum chemistry program used as engine: QChem, TeraChem, Psi4
 - Optimizer: geomeTRIC or the built-in optimizer from the QM program
 - Distributed: run optimization locally or distribute them using ``cctools.work_queue``

geomeTRIC + Psi4
+++++++++++++++++++++++++++++++++++++++
 - Location: torsiondrive_examples/examples/hooh-1d/psi4/run_local/geomeTRIC/
 - Run command: ``torsiondrive-launch input.dat dihedrals.txt -g 15 -e psi4 -v``
 - Output log: scan.log
 - Energy plot can be generated using ``torsiondrive-plot1d``
    .. image:: media/1d_plot.jpeg
        :width: 500px
        :align: center


2-D Examples
------------
2-D torsion scans are relatively expensive. Therefore it is recommended to use a cheap QM method, or use
the distributed method calling `cctools.work_queue`.

geomeTRIC + Psi4 distributed
+++++++++++++++++++++++++++++++++++++++
 - Location: torsiondrive_examples/examples/propanol-2d/work_queue_qchem_geomeTRIC/
 - Run command: ``torsiondrive-launch qc.in dihedrals.txt -g 15 -e qchem --wq_port 50124 -v 2>worker.log``
 - Two dihedrals are specified in input ``dihedrals.txt`` to create a 2-D scan::

    # dihedral definition by atom indices starting from 1
    # i     j     k     l
      1     2     7     11
      2     7     11    12

 - Output log: scan.log
 - Energy heatmap can be generated using ``torsiondrive-plot2d``
    .. image:: media/2d_heatmap.jpeg
        :width: 800px
        :align: center


range limited scan
+++++++++++++++++++++++++++++++++++++++
 - Location: torsiondrive_examples/examples/range_limited_split/
 - Run command: ``torsiondrive-launch qc.in dihedrals.txt -g 15 30 -e qchem -v --wq_port 50124 2>worker.log``
 - Input dihedrals.txt::

    # dihedral definition by atom indices starting from 1
    # i     j     k     l      (range_low)     (range_high)
      1     2     7     11        -60              60
      2     7     11    12        150             330

 - Output log: scan.log
 - Energy heatmap can be generated using ``torsiondrive-plot2d``
    .. image:: media/2d_heatmap_limited.jpeg
        :width: 800px
        :align: center
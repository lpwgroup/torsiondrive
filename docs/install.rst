Install TorsionDrive
====================

You can install `torsiondrive` with ``conda``, with ``pip``, or by installing from source.

Conda
-----

You can update torsiondrive using `conda <https://www.anaconda.com/download/>`_::

    conda install torsiondrive -c conda-forge

This installs torsiondrive and its dependancies.

The torsiondrive package is maintained on the
`conda-forge channel <https://conda-forge.github.io/>`_.


Pip
---

To install torsiondrive with ``pip`` ::

    pip install torsiondrive

Install from Source
-------------------

To install qcfractal from source, clone the repository from `github
<https://github.com/lpwgroup/torsiondrive>`_::

    git clone https://github.com/lpwgroup/torsiondrive.git
    cd torsiondrive
    python setup.py install

or use ``pip`` for a local install::

    pip install -e .

It is recommended to setup a testing environment using ``conda``. This can be accomplished by::

    cd torsiondrive
    python devtools/scripts/conda_env.py -n=td_test -p=3.7 devtools/conda-envs/psi.yaml


Test
----

Test torsiondrive with ``py.test``::

    cd torsiondrive
    py.test


Installation of cctools
------------------------
The library ``cctools.work_queue`` is utilized to provide distributed computing feature in TorsionDrive.
https://github.com/cooperative-computing-lab/cctools

Installation of ``cctools`` is provided separately. A convenient bash script has been made to simplify the process::

    $bash torsiondrive/devtools/travis-ci/install-cctools.sh
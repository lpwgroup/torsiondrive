#!/bin/bash

# Download latest version from website.
echo "Downloading source."
version="7.0.14"
cctools="cctools/$version"
rm -rf cctools*
wget https://github.com/lpwgroup/cctools/archive/v${version}.tar.gz
echo "Extracting archive."
tar xzf v${version}.tar.gz
cd cctools-${version}

# Increase all sorts of timeouts.
sed -i s/"timeout = 5;"/"timeout = 7200;"/g work_queue/src/*.c
sed -i s/"timeout = 10;"/"timeout = 7200;"/g work_queue/src/*.c
sed -i s/"timeout = 15;"/"timeout = 7200;"/g work_queue/src/*.c
sed -i s/"foreman_transfer_timeout = 3600"/"foreman_transfer_timeout = 86400"/g work_queue/src/work_queue.c
sed -i s/"long_timeout = 3600"/"long_timeout = 86400"/g work_queue/src/work_queue.c

# Disable perl
sed -i s/"config_perl_path=auto"/"config_perl_path=no"/g configure

# Disable globus
sed -i s/"config_globus_path=auto"/"config_globus_path=no"/g configure

#----
# Provide install prefix for cctools as well as
# locations of Swig and Python packages (i.e. the
# executable itself is inside the bin subdirectory).
#
# This is to ensure that we can call the correct
# versions of Python and Swig since the version
# installed for the OS might be too old.
#----
prefix=$HOME/opt
swgpath=$(dirname $(dirname $(which swig)))
pypath=$(dirname $(dirname $(which python)))

# Create these directories if they don't exist.
mkdir -p $prefix
mkdir -p $swgpath
mkdir -p $pypath

if [ ! -d $prefix ] ; then
    echo "Warning: Installation directory $prefix does not exist."
    read
fi

if [ ! -f $swgpath/bin/swig ] ; then
    echo "Warning: $swgpath does not point to Swig."
    read
fi

if [ ! -f $pypath/bin/python ] ; then
    echo "Warning: $pypath does not point to Python."
    read
fi

#----
# The following assumes that cctools will be installed into $HOME/opt
# and Python lives in $HOME/local.
#----
# Configure, make, make install.
# check python version
PYTHON_VERSION=`python -c 'import sys; print(sys.version_info[0])'`
if [ "$PYTHON_VERSION" -eq "3" ]
then
    ./configure --prefix $prefix/$cctools --with-python3-path $pypath --with-python-path no --with-swig-path $swgpath --with-perl-path no --with-globus-path no
else
    ./configure --prefix $prefix/$cctools --with-python-path $pypath --with-swig-path $swgpath
fi
make && make install && cd work_queue && make install

#----
# Make symbolic link from installed version to plain "cctools" folder.
# This allows you to add $HOME/cctools/bin to your PATH.
#----
cd $prefix/
rm -f cctools
ln -s $cctools cctools
cd cctools/bin
for i in wq_submit_workers.common sge_submit_workers torque_submit_workers slurm_submit_workers ; do
    if [ -f $HOME/etc/work_queue/$i ] ; then
        echo "Replacing $i with LP's custom version"
        mv $i $i.bak
        ln -s $HOME/etc/work_queue/$i .
    fi
done
cd ../..

# Install Python module.
PYTHON_SITEPACKAGES=`python -c "import site; print(site.getsitepackages()[0])"`
PNAME=$(basename $(dirname $PYTHON_SITEPACKAGES))
echo "Before installing Python module, will remove these files"
echo "from $PYTHON_SITEPACKAGES"
rm -f $PYTHON_SITEPACKAGES/*work_queue*
cp -r $prefix/$cctools/lib/$PNAME/site-packages/* $PYTHON_SITEPACKAGES
ls $PYTHON_SITEPACKAGES/*work_queue*
echo "Python module installed"

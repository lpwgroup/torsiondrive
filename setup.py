"""
torsiondrive
Dihedral scanner with wavefront propagation
"""
from setuptools import setup, find_packages
import versioneer

DOCLINES = __doc__.split("\n")

setup(
    name='torsiondrive',
    author='Yudong Qiu, Lee-Ping Wang',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    entry_points={'console_scripts': [
        'torsiondrive-launch = torsiondrive.launch:main',
        'torsiondrive-api = torsiondrive.td_api:main',
        'torsiondrive-plot1d = torsiondrive.tools.plot_1d:main',
        'torsiondrive-plot2d = torsiondrive.tools.plot_2d:main',
    ]},
    url='https://github.com/lpwgroup/torsiondrive',
    install_requires=[
        'numpy>=1.11',
        'geometric>=0.9.7'
    ]
)

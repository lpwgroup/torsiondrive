"""
torsiondrive
Dihedral scanner with wavefront propagation
"""
from setuptools import setup
import versioneer

DOCLINES = __doc__.split("\n")

setup(
    name='torsiondrive',
    author='Yudong Qiu, Lee-Ping Wang',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['torsiondrive', "torsiondrive.tests"],
    entry_points={'console_scripts': [
        'torsiondrive-launch = torsiondrive.launch:main',
        'torsiondrive-api = torsiondrive.td_api:main',
    ]},
    url='https://github.com/lpwgroup/torsiondrive',
    install_requires=[
        'numpy>=1.11',
        'geometric'
    ]
)

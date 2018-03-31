"""
crank
Dihedral scanner with wavefront propagation
"""
from setuptools import setup
import versioneer

DOCLINES = __doc__.split("\n")

setup(
    name='crank',
    author='Yudong Qiu, Lee-Ping Wang',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['crank', "crank.tests"],
    entry_points={'console_scripts': [
        'crank-launch = crank.launch:main',
        'crank-api = crank.crankAPI:main',
    ]},
    url='https://github.com/lpwgroup/crank',
    install_requires=[
        'numpy>=1.11',
        'geometric'
    ]
)


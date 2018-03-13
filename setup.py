from setuptools import setup

setup(
    name='crank',
    description='Dihedral scanner with wave propagation',
    url='https://github.com/lpwgroup/crank',
    author='Yudong Qiu, Lee-Ping Wang',
    packages=['crank'],
    entry_points={'console_scripts': [
        'crank-launch = crank.launch:main',
        'crank-api = crank.crankAPI:main',
    ]},
    install_requires=[
        'numpy>=1.11',
        'geometric'
    ])
#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np

from geometric.molecule import Molecule

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def read_scan_xyz(filename):
    """ Read the scan xyz file and return a dictionary of {grid_id: energy} """
    res = {}
    m = Molecule(filename)
    for line in m.comms:
        ls = line.split()
        # read the second element as grid id
        grid_id_str = ls[1]
        assert grid_id_str[0] == '(' and grid_id_str[-1] == ')'
        grid_id = tuple(int(s) for s in grid_id_str[1:-1].split(',') if s)
        # read the last element as energy
        energy = float(ls[-1])
        res[grid_id] = energy
    return res

def plot_1d_curve(data, filename):
    if not data:
        print('data is empty, returning')
        return
    sorted_grid_id = sorted(data.keys())
    assert len(sorted_grid_id[0]) == 1, "only 1d grid plot is supported"

    dihedrals = [grid_id[0] for grid_id in sorted_grid_id]
    energies = [data[grid_id] for grid_id in sorted_grid_id]
    # convert energy to kcal/mol
    energies = np.array(energies) * 627.509
    # convert absolute energy to relative energy
    energies -= energies.min()
    # plot
    plt.Figure()
    plt.plot(dihedrals, energies, '-o')
    #plt.xticks(dihedrals[1::2])
    plt.xlabel('Dihedral Angle [Degree]')
    plt.ylabel('Relative Energy [kcal/mol]')
    plt.title("Torsion Energy Profile")
    plt.savefig(filename)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('xyz', nargs='*', default=['scan.xyz'], help='Input scan xyz file')
    args = parser.parse_args()

    for f in args.xyz:
        data = read_scan_xyz(f)
        pdf_filename = os.path.splitext(f)[0] + '.pdf'
        plot_1d_curve(data, pdf_filename)
        print("Generated %s" % pdf_filename)

if __name__ == '__main__':
    main()

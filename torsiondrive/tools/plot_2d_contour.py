#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_data_from_scan_xyz(filename):
    """ Read the dihedral information and energy from scan.xyz """
    with open(filename) as f:
        lines = f.readlines()
    n_atoms = int(lines[0])
    comment_lines = lines[1::n_atoms+2]
    grid_data = dict()
    for line in comment_lines:
        ls = line.strip().split()
        assert ls[0] == 'Dihedral' and ls[-2] == 'Energy', line
        grid_energy = float(ls[-1])
        grid_coord = []
        for i in range(1, len(ls) - 2):
            c = int(ls[i].replace('(', '').replace(',','').replace(')',''))
            grid_coord.append(c)
        grid_data[tuple(grid_coord)] = grid_energy
    return grid_data

def plot_grid_contour(grid_data, method='imshow', vmax=None):
    """ Plot grid data as a contour map """
    grid_size = int(len(grid_data)**0.5)
    grid_spacing = int(360 / grid_size)
    print(f"grid_size: {grid_size}  grid_spacing: {grid_spacing}")
    x_array = np.arange(-180, 180, grid_spacing, dtype=int) + grid_spacing
    y_array = np.arange(-180, 180, grid_spacing, dtype=int) + grid_spacing
    z_mat = np.zeros((grid_size, grid_size))
    for i, x in enumerate(x_array):
        for j, y in enumerate(y_array):
            z_mat[i, j] = grid_data.get((x,y), np.nan)
    # convert abs energies to relative energies
    z_mat = (z_mat - z_mat.min()) * 627.509
    if method == 'imshow':
        plt.imshow(z_mat, vmax=vmax, cmap='rainbow', origin='lower', extent=(-165, 180, -165, 180))
    elif method == 'contourf':
        plt.contourf(x_array, y_array, z_mat, vmax=vmax, antialiased=True, cmap='rainbow')
    plt.colorbar()
    plt.xticks(x_array[1::2], x_array[1::2])
    plt.yticks(y_array[1::2], y_array[1::2])
    cs = plt.contour(x_array, y_array, z_mat, vmax=vmax, antialiased=True, colors='black')
    plt.clabel(cs, fontsize=9, inline=1)
    plt.savefig('contour.pdf')
    print("Plot saved as contour.pdf")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help='scan.xyz file from torsionscan launch')
    parser.add_argument("-m", "--method", choices=['contourf', 'imshow'], default='imshow', help='method to color background')
    parser.add_argument("--vmax", type=float, help='max value of heat map')
    args = parser.parse_args()

    grid_data = load_data_from_scan_xyz(args.infile)
    plot_grid_contour(grid_data, method=args.method, vmax=args.vmax)

if __name__ == '__main__':
    main()





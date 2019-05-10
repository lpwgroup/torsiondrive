# Useful utilities for parsing output files like scan.xyz

import math
from geometric.molecule import Molecule

def read_scan_xyz(filename):
    """
    Parse the scan xyz file into a dictionary

    Parameters
    ----------
    filename: str
        path to the scan.xyz file generated by torsiondrive

    Returns
    -------
    grid_data: dict
        A dictionary of {grid_id: energy} """
    grid_data = {}
    m = Molecule(filename)
    for line in m.comms:
        # parse comment line, find grid id between "(" and ")"
        try:
            left_p_idx = line.index('(')
            right_p_idx = line.index(')')
        except ValueError:
            print("Grid id in (XX, XX) format not found in file")
            raise
        grid_id_str = line[left_p_idx+1: right_p_idx]
        grid_id = tuple(int(s) for s in grid_id_str.split(',') if s)
        # read the last element as energy
        ls = line.rsplit(maxsplit=1)
        energy = float(ls[-1])
        grid_data[grid_id] = energy
    return grid_data

def find_grid_spacing(grid_id_list):
    """ Find the largest possible grid spacing for one dimension grid id list """
    if not grid_id_list: return None
    assert all(-180 < grid_id <= 180 for grid_id in grid_id_list), f"grid id out of range (-180, 180]: {grid_id_list}"
    if len(grid_id_list) == 1:
        # find the largest possible grid spacing if only one data is available this direction
        # The answer is the largest divisor of
        # a: grid range 360
        # b: distance from grid_id to -180
        grid_id = grid_id_list[0]
        res = math.gcd(360, grid_id+180)
    else:
        n = len(grid_id_list)
        grid_id_list = sorted(grid_id_list)
        res = 360
        for i in range(n):
            gid = grid_id_list[i]
            spacing = math.gcd(360, gid+180)
            res = math.gcd(res, spacing)
            if i < n-1:
                step = grid_id_list[i+1] - gid
                res = math.gcd(res, step)
    return res
# crankServerAPI.py
# This file is expected to be loaded by the server
# It contains functions for the server to generate inputs for crankAPI and parse outputs of crankAPI

import collections
import copy


def create_initial_api_input(dihedrals, grid_spacing, elements, init_coords):
    """ Create the initial input dictionary for crank-api """
    return {
        'dihedrals': dihedrals,
        'grid_spacing': grid_spacing,
        'elements': elements,
        'init_coords': init_coords,
        'grid_status': collections.defaultdict(list)
    }

def collect_lowest_energies(crank_state):
    """ Find the lowest energies for each dihedral grid from crank_state """
    lowest_energies = collections.defaultdict(lambda: float('inf'))
    for grid_id_str, job_result_tuple_list in crank_state['grid_status'].items():
        grid_id = tuple(int(i) for i in grid_id_str.split(','))
        for start_geo, end_geo, end_energy in job_result_tuple_list:
            lowest_energies[grid_id] = min(lowest_energies[grid_id], end_energy)
    return lowest_energies

def update_crank_state(crank_state, job_results):
    updated_crank_state = copy.deepcopy(crank_state)
    for grid_id_str, job_result_tuple_list in job_results.items():
        updated_crank_state['grid_status'][grid_id_str] += job_result_tuple_list
    return updated_crank_state

def gridIDStr_to_dihedralValues(grid_id_str):
    return tuple(int(i) for i in grid_id_str.split(','))

# coding: utf-8
from geometric.nifty import ang2bohr
import numpy as np
import json
d = json.load(open('current_state.json'))
c = np.array(d['init_coords'][0])
c *= ang2bohr
d['init_coords'][0] = c.tolist()
new_grid_status = dict()
for g, results in d['grid_status'].items():
    new_results = []
    for start_geo, end_geo, energy in results:
        new_start_geo = (np.array(start_geo) * ang2bohr).tolist()
        new_end_geo = (np.array(end_geo) * ang2bohr).tolist()
        new_results.append([new_start_geo, new_end_geo, energy])
    new_grid_status[g] = new_results
d['grid_status'] = new_grid_status
json.dump(d, open('test.json', 'w'))

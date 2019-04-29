#------------------------
#  extra_constraints.py
#------------------------
from geometric.nifty import uncommadash, commadash

def make_constraints_dict(constraints_string):
    """ Create an ordered dictionary with constraints specification, consistent with geomeTRIC

    Parameters
    ----------
    constraints_string: str
        String-formatted constraint specification consistent with geomeTRIC constraints.txt

    Returns
    -------
    constraints_dict: dict
        A dictionary contains the definition of the extra constraints. The format is consistant with the JSON interface
        of geomeTRIC.

    Examples
    --------
    >>> make_constraints_dict(r"$freeze\\nxyz 1-3\\n$set\\nangle 2 1 5 110.0")
    {
        'freeze': [{
            'type': 'xyz',
            'indices': [0, 1, 2]
        }],
        'set': [{
            'type': 'angle',
            'indices': [1, 0, 4],
            'value': 110.0}]
    }

    Notes
    -----
    1. Only constraints of type "freeze" and "set" are supported, since extra "scan" is undefined with torsiondrive scan.
    2. Four attributes are allowed to be constrained: 'distance', 'angle', 'dihedral', 'xyz'
    3. The input string is one-indexed, the output dictionary is zero-indexed.
    4. For "xyz", dashed inputs like "1-3,7-9" (no space) is allowed, and will be converted to [0,1,2,6,7,8].
    """
    constraints_mode = None
    constraints_dict = {'freeze': [], 'set': []}
    for line in constraints_string.split('\n'):
        # Ignore anything after a comment
        line = line.split('#',1)[0].lower().strip()
        if len(line) == 0: continue
        if line.startswith('$'):
            if line == '$freeze':
                constraints_mode = 'freeze'
            elif line == '$set':
                constraints_mode = 'set'
            elif line == '$end':
                constraints_mode = None
            elif line == '$scan':
                raise ValueError(f'Additional $scan constraints are not allowed')
            else:
                raise ValueError(f'Line {line}\nUnrecognized token starting with $')
        else:
            if constraints_mode == None:
                raise ValueError(f'Trying to read the constraint line\n\n{line}\n\n, but constraint mode is not set')
            elif constraints_mode == 'freeze':
                ls = line.split()
                ctype = ls[0]
                if ctype not in ['distance','angle','dihedral','xyz']:
                    raise ValueError(f'Line {line}\nOnly distance, angle, and dihedral, xyz constraints are supported')
                indices = [int(i)-1 for i in ls[1:]] if ctype != 'xyz' else uncommadash(ls[1])
                assert all(i >= 0 for i in indices), f'Invalid atom index in line\n{line}\n, one-indexed should start from 1'
                spec_dict = { 'type': ctype, 'indices': indices }
                constraints_dict[constraints_mode].append(spec_dict)
            elif constraints_mode == 'set':
                ls = line.split()
                ctype = ls[0]
                # we don't support setting xyz here because it's confusing
                if ctype not in ['distance','angle','dihedral']:
                    raise ValueError(f'Line {line}\nOnly distance, angle, and dihedral constraints are supported by Set')
                indices = [int(i)-1 for i in ls[1:-1]]
                assert all(i >= 0 for i in indices), f'Invalid atom index in line {line}, one-indexed should start from 1'
                value = float(ls[-1])
                spec_dict = { 'type': ctype, 'indices': indices, 'value': value }
                constraints_dict[constraints_mode].append(spec_dict)
            else:
                raise ValueError(f"Line {line}\nConstraints mode {constraints_mode} is not supported")
    return constraints_dict

def check_conflict_constraints(constraints_dict, dihedral_idxs):
    """
    Utility function to check if any extra constraints in constraints_dict is conflict with the scanning dihedrals
    """
    distinct_dihedrals = set(tuple(min(d, d[::-1])) for d in dihedral_idxs)
    distinct_dihedral_centers = set(tuple(min(d[1:3], d[2:0:-1])) for d in dihedral_idxs)
    for constraits_list in constraints_dict.values():
        for spec_dict in constraits_list:
            ctype, indices = spec_dict['type'], spec_dict['indices']
            if ctype == 'dihedral':
                if tuple(min(indices, indices[::-1])) in distinct_dihedrals:
                    # the same dihedral appears in extra_constraints
                    raise ValueError(f"Conflict dihedral constraints found in:\n{spec_dict}\n with {dihedral_idxs}")
                elif tuple(min(indices[1:3], indices[2:0:-1])) in distinct_dihedral_centers:
                    # Lee-Ping pointed out that geomeTRIC have issue if two dihedral constraints share the same center atoms
                    raise ValueError(f"Extra dihedral constraint in:\n{spec_dict}\n share the same center atoms with {dihedral_idxs}")
            elif ctype == 'xyz':
                indices_set = set(indices)
                # check if all 4 indices defining a dihedral are froze
                if any(set(d).issubset(indices_set) for d in distinct_dihedrals):
                    raise ValueError(f"Conflict xyz constraints found in:\n{spec_dict}\n with {dihedral_idxs}")

def build_geometric_constraint_string(constraints_dict, dihedral_idx_values=None):
    """
    Build the geomeTRIC constraint string with constraints_dict and a set of dihedral_idx_values

    Parameters
    ----------
    constraints_dict: Dict
        constraints dict built by make_constraints_dict() function

    dihedral_idx_values: List[List[d1, d2, d3, d4, v]]
        A list containing the definition of dihedrals and their values
        Example: [(0,1,2,3,90.0), (1,2,3,4,100.0)]

    Returns
    -------
    constraints_string: string
        A string with multiple lines, to be used as the geomeTRIC constraints.txt
    """
    constraints_string = ''
    # write the "$freeze" section
    spec_list = constraints_dict.get('freeze',[])
    if len(spec_list) > 0:
        constraints_string += '$freeze\n'
        for spec_dict in spec_list:
            ctype, indices = spec_dict['type'], spec_dict['indices']
            if ctype == 'xyz':
                constraints_string += f'xyz {commadash(indices)}' + '\n'
            else:
                constraints_string += f'{ctype} ' + ' '.join(map(str, [i+1 for i in indices])) + '\n'
    # write the "$set" section
    set_section_open = False
    spec_list = constraints_dict.get('set',[])
    if len(spec_list) > 0:
        constraints_string += '$set\n'
        set_section_open = True
        for spec_dict in spec_list:
            ctype, indices, value = spec_dict['type'], spec_dict['indices'], spec_dict['value']
            constraints_string += f'{ctype} ' + ' '.join(map(str, [i+1 for i in indices])) + f' {float(value)}\n'
    # write dihedral_idx_values as constraints
    if dihedral_idx_values is not None:
        if set_section_open is False:
            # write the $set head if not written yet
            constraints_string += '$set\n'
        for d1, d2, d3, d4, v in dihedral_idx_values:
            constraints_string += f"dihedral {d1+1} {d2+1} {d3+1} {d4+1} {float(v)}\n"
    return constraints_string

def build_terachem_constraint_string(constraints_dict, dihedral_idx_values=None):
    """
    Build the TeraChem constraint string with constraints_dict and a set of dihedral_idx_values

    Parameters
    ----------
    constraints_dict: Dict
        constraints dict built by make_constraints_dict() function

    dihedral_idx_values: List[List[d1, d2, d3, d4, v]]
        A list containing the definition of dihedrals and their values
        Example: [(0,1,2,3,90), (1,2,3,4,100)]

    Returns
    -------
    constraints_string: string
        A string with multiple lines, to be used as the TeraChem constraints format
    """
    constraints_string = '\n'
    # write the "$constraint_freeze" section
    spec_list = constraints_dict.get('freeze',[])
    if len(spec_list) > 0:
        constraints_string += '$constraint_freeze\n'
        for spec_dict in spec_list:
            ctype, indices = spec_dict['type'], spec_dict['indices']
            if ctype == 'xyz':
                constraints_string += f'xyz {commadash(indices)}' + '\n'
            else:
                # TeraChem only take "bond" keyword
                if ctype == 'distance': ctype = 'bond'
                constraints_string += f'{ctype} ' + '_'.join(map(str, [i+1 for i in indices])) + '\n'
        constraints_string += '$end\n\n'
    # write the "$constraint_set" section
    set_section_open = False
    spec_list = constraints_dict.get('set',[])
    if len(spec_list) > 0:
        constraints_string += '$constraint_set\n'
        set_section_open = True
        for spec_dict in spec_list:
            ctype, indices, value = spec_dict['type'], spec_dict['indices'], spec_dict['value']
            # TeraChem only take "bond" keyword
            if ctype == 'distance': ctype = 'bond'
            constraints_string += f'{ctype} {float(value)} ' + '_'.join(map(str, [i+1 for i in indices])) + '\n'
    # write dihedral_idx_values as constraints
    if dihedral_idx_values is not None:
        if set_section_open is False:
            # write the $set head if not written yet
            constraints_string += '$constraint_set\n'
            set_section_open = True
        for d1, d2, d3, d4, v in dihedral_idx_values:
            constraints_string += f"dihedral {float(v)} {d1+1}_{d2+1}_{d3+1}_{d4+1}\n"
    if set_section_open is True:
        constraints_string += '$end\n\n'
    return constraints_string

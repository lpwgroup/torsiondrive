#------------------------
#  extra_constraints.py
#------------------------
from geometric.nifty import uncommadash

def constraint_match(cons_type, cons_spec, indices):
    """ Check to see if a string-specified constraint matches a set of provided zero-based indices. """
    s = cons_spec.split()
    if cons_type in ['bond', 'distance']:
        sMatch = [int(i)-1 for i in s[:2]]
        if len(indices) != 2: return False
        elif tuple(sMatch) == tuple(indices): return True
        elif tuple(sMatch[::-1]) == tuple(indices): return True
        else: return False
    elif cons_type == 'angle':
        sMatch = [int(i)-1 for i in s[:3]]
        if len(indices) != 3: return False
        elif tuple(sMatch) == tuple(indices): return True
        elif tuple(sMatch[::-1]) == tuple(indices): return True
        else: return False
    elif cons_type == 'dihedral':
        # Dihedrals should "match" if the two middle atoms are the same
        sMatch = [int(i)-1 for i in s[:4]]
        if len(indices) != 4: return False
        elif (sMatch[1], sMatch[2]) == (indices[1], indices[2]): return True
        elif (sMatch[1], sMatch[2]) == (indices[2], indices[1]): return True
        else: return False
    elif cons_type == 'xyz':
        sMatch = uncommadash(cons_spec)
        if sMatch == sorted(indices): return True
        elif len(indices) == 4 and indices[0] in sMatch and indices[3] in sMatch: return True
        else: return False
    else:
        raise RuntimeError("Problem detected with user-supplied extra constraints")

def make_constraints_dict(constraints_string, exclude=[]):
    """ Create an ordered dictionary with constraints specification, consistent with geomeTRIC

    Parameters
    ----------
    constraints_string: str
        String-formatted constraint specification consistent with geomeTRIC constraints.txt
    exclude: list, optional
        A list of zero-based constraint atom indices (e.g. dihedrals to be scanned over)
        which should not be included in this dictionary

    Returns
    -------
    constraints_dict: dict
        A dictionary contains the definition of the extra constraints
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
                raise RuntimeError('Additional $scan constraints are not allowed')
            else:
                raise RuntimeError('Unrecognized token starting with $')
        else:
            if constraints_mode == None:
                print('ERROR: ', line)
                raise RuntimeError('Trying to read the above constraint line, but constraint mode is not set')
            else:
                spec_tuple = tuple(line.split())
                if spec_tuple[0] not in ['bond','distance','angle','dihedral','xyz']:
                    raise RuntimeError('Only bond, angle, and dihedral, xyz constraints are supported')
                if any([constraint_match(spec_tuple[0], ' '.join(spec_tuple[1:]), excl) for excl in exclude]): continue
                constraints_dict[constraints_mode].append(spec_tuple)
    return constraints_dict
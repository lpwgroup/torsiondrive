"""
Unit and regression test for the torsiondrive extra constraints feature
"""

import pytest
from torsiondrive.extra_constraints import make_constraints_dict, check_conflict_constraints, \
    build_geometric_constraint_string, build_terachem_constraint_string


def test_make_constraints_dict():
    '''
    Unit test for torsiondrive.qm_engine.make_constraints_dict() function
    '''
    constraits_string = '''
    $freeze
    xyz 1-3,6-7
    distance 1 5
    $set
    distance 1 4 1.0
    angle 4 1 2 30.0
    dihedral 1 2 3 4 60.0
    '''
    constraints_dict = make_constraints_dict(constraits_string)
    # test the freeze constraints
    spec_list = constraints_dict['freeze']
    assert len(spec_list) == 2
    assert spec_list[0]['type'] == 'xyz'
    assert spec_list[0]['indices'] == [0, 1, 2, 5, 6]
    assert spec_list[1]['type'] == 'distance'
    assert spec_list[1]['indices'] == [0, 4]
    # test the set constraints
    spec_list = constraints_dict['set']
    assert len(spec_list) == 3
    assert spec_list[0]['type'] == 'distance'
    assert spec_list[0]['indices'] == [0, 3]
    assert spec_list[0]['value'] == 1.0
    assert spec_list[1]['type'] == 'angle'
    assert spec_list[1]['indices'] == [3, 0, 1]
    assert spec_list[1]['value'] == 30.0
    assert spec_list[2]['type'] == 'dihedral'
    assert spec_list[2]['indices'] == [0, 1, 2 ,3]
    assert spec_list[2]['value'] == 60.0
    # test error handling for wrong inputs
    # scan not supported
    wrong_constraits_string = '''
    $scan
    distance 1 2 1.0 2.0 10
    '''
    with pytest.raises(ValueError):
        make_constraints_dict(wrong_constraits_string)
    # $unknown not supported
    wrong_constraits_string = '''
    $unknown
    distance 1 2 1.0 2.0 10
    '''
    with pytest.raises(ValueError):
        make_constraints_dict(wrong_constraits_string)
    # rotation not supported
    wrong_constraits_string = '''
    $set
    rotation 1-4 10.0
    '''
    with pytest.raises(ValueError):
        make_constraints_dict(wrong_constraits_string)
    # freezing x not supported
    wrong_constraits_string = '''
    $freeze
    x 1 1.0
    '''
    with pytest.raises(ValueError):
        make_constraints_dict(wrong_constraits_string)
    # missing $set
    wrong_constraits_string = '''
    distance 1 2 1.0
    '''
    with pytest.raises(ValueError):
        make_constraints_dict(wrong_constraits_string)
    # setting xyz not supported
    wrong_constraits_string = '''
    $set
    xyz 1-3 1.0 2.0 3.0
    '''
    with pytest.raises(ValueError):
        make_constraints_dict(wrong_constraits_string)
    # wrong index
    wrong_constraits_string = '''
    $set
    distance 0 1 1.0
    '''
    with pytest.raises(AssertionError):
        make_constraints_dict(wrong_constraits_string)

def test_check_conflict_constraints():
    """
    Test the check_conflict_constraints() function

    Notes
    -----
    Atom indices in constraits_string is 1-indexed
    Atom indices in dihedral_idxs is 0-indexed
    """
    constraits_string = '''
    $freeze
    xyz 1-3,6-7
    distance 1 5
    $set
    distance 1 4 1.0
    angle 4 1 2 30.0
    dihedral 1 2 3 4 60.0
    '''
    constraints_dict = make_constraints_dict(constraits_string)
    # empty check
    check_conflict_constraints(constraints_dict, [])
    # test valid constraints_dict when not conflicting
    dihedral_idxs = [[1,2,3,4], [2,3,4,5]]
    check_conflict_constraints(constraints_dict, dihedral_idxs)
    # test conflicting
    with pytest.raises(ValueError):
        # dihedral conflict (same as scanning)
        check_conflict_constraints(constraints_dict, [[0,1,2,3]])
    with pytest.raises(ValueError):
        # dihedral conflict (share the same center atoms 1, 2)
        check_conflict_constraints(constraints_dict, [[5,2,1,6]])
    with pytest.raises(ValueError):
        # xyz conflict
        check_conflict_constraints(constraints_dict, [[0,1,2,5]])

def test_build_geometric_constraint_string():
    """
    Test build_geometric_constraint_string() function

    Notes
    -----
    Atom indices in constraits_string is 1-indexed
    Atom indices in dihedral_idxs is 0-indexed
    """
    constraints_dict = {
        'freeze': [
            {
                'type': 'xyz',
                'indices': [0, 1, 2, 5, 6],
            },
            {
                'type': 'distance',
                'indices': [0, 4],
            }
        ],
        'set': [
            {
                'type': 'distance',
                'indices': [0, 3],
                'value': 1.0,
            },
            {
                'type': 'angle',
                'indices': [1, 0, 3],
                'value': 30.0,
            },
            {
                'type': 'dihedral',
                'indices': [0, 1, 2, 3],
                'value': 60.0,
            },
        ]
    }
    constraints_string = build_geometric_constraint_string(constraints_dict)
    # validate the constraints string
    assert constraints_string.strip() == '\n'.join(['$freeze', 'xyz 1-3,6-7', 'distance 1 5', '$set', 'distance 1 4 1.0',
        'angle 2 1 4 30.0', 'dihedral 1 2 3 4 60.0'])
    # test with dihedral_idx_values
    dihedral_idx_values=[(1,2,3,4,90.0), (2,3,4,5,120.0)]
    constraints_string2 = build_geometric_constraint_string(constraints_dict, dihedral_idx_values=dihedral_idx_values)
    assert constraints_string2.strip() == '\n'.join(['$freeze', 'xyz 1-3,6-7', 'distance 1 5', '$set', 'distance 1 4 1.0',
        'angle 2 1 4 30.0', 'dihedral 1 2 3 4 60.0', 'dihedral 2 3 4 5 90.0', 'dihedral 3 4 5 6 120.0'])

def test_build_terachem_constraint_string():
    """
    Test build_geometric_constraint_string() function

    Notes
    -----
    Atom indices in constraits_string is 1-indexed
    Atom indices in dihedral_idxs is 0-indexed
    """
    constraints_dict = {
        'freeze': [
            {
                'type': 'xyz',
                'indices': [0, 1, 2, 5, 6],
            },
            {
                'type': 'distance',
                'indices': [0, 4],
            }
        ],
        'set': [
            {
                'type': 'distance',
                'indices': [0, 3],
                'value': 1.0,
            },
            {
                'type': 'angle',
                'indices': [1, 0, 3],
                'value': 30.0,
            },
            {
                'type': 'dihedral',
                'indices': [0, 1, 2, 3],
                'value': 60.0,
            },
        ]
    }
    constraints_string = build_terachem_constraint_string(constraints_dict)
    # validate the constraints string
    assert constraints_string.strip() == '\n'.join(['$constraint_freeze', 'xyz 1-3,6-7', 'bond 1_5', '$end\n',
        '$constraint_set', 'bond 1.0 1_4', 'angle 30.0 2_1_4', 'dihedral 60.0 1_2_3_4', '$end'])
    # test with dihedral_idx_values
    dihedral_idx_values=[(1,2,3,4,90.0), (2,3,4,5,120.0)]
    constraints_string2 = build_terachem_constraint_string(constraints_dict, dihedral_idx_values=dihedral_idx_values)
    assert constraints_string2.strip() == '\n'.join(['$constraint_freeze', 'xyz 1-3,6-7', 'bond 1_5', '$end\n',
        '$constraint_set', 'bond 1.0 1_4', 'angle 30.0 2_1_4', 'dihedral 60.0 1_2_3_4',
        'dihedral 90.0 2_3_4_5', 'dihedral 120.0 3_4_5_6', '$end'])

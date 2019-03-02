"""
Unit and regression test for the torsiondrive extra constraints feature
"""

import pytest
from torsiondrive.extra_constraints import make_constraints_dict

def test_make_constraints_dict():
    '''
    Unit test for torsiondrive.qm_engine.make_constraints_dict() function
    '''
    constraits_string = '''
    $set
    bond 1 2 1.0
    '''
    constraints_dict = make_constraints_dict(constraits_string)
    assert len(constraints_dict['set']) == 1
    spec_dict = constraints_dict['set'][0]
    assert spec_dict['type'] == ('bond', [0,1])
    assert spec_dict['value'] == 1.0

"""
Unit and regression test for the torsiondrive extra constraints feature
"""

import pytest
import torsiondrive

def test_make_constraints_dict():
    '''
    Unit test for torsiondrive.qm_engine.make_constraints_dict() function
    '''
    constraits_string = '''
    $set
    bond 1 2 1.0
    '''
    constraints_dict = torsiondrive.qm_engine.make_constraints_dict(constraits_string)
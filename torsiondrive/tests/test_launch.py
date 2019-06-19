"""
Unit and regression test for the torsiondrive.launch module
"""

import pytest
from torsiondrive.launch import load_dihedralfile, create_engine

def test_load_dihedralfile_basic(tmpdir):
    tmpdir.chdir()
    fn = 'dihedrals.txt'
    # basic test loading one dihedral
    dihedral_str = '''
    #i j k l
     1 2 3 4
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    assert dihedral_idxs ==  [[0,1,2,3]]
    assert dihedral_ranges == []
    # basic test loading 2 dihedrals
    dihedral_str = '''
    #i j k l
     1 2 3 4
     2 3 4 5
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    assert dihedral_idxs ==  [[0,1,2,3], [1,2,3,4]]
    assert dihedral_ranges == []
    # test with wrong number of index
    dihedral_str = '''
    #i j k l
     1 2 3
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    with pytest.raises(ValueError):
        dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    # test wrong index 0 when default is one-based
    dihedral_str = '''
    #i j k l
     0 1 2 3
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    with pytest.raises(AssertionError):
        dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)

def test_load_dihedralfile_0_1_numbering(tmpdir):
    tmpdir.chdir()
    fn = 'dihedrals.txt'
    # test loading with zero_based_numbering option
    dihedral_str = '''
    #i j k l
     1 2 3 4
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    dihedral_idxs, dihedral_ranges = load_dihedralfile(fn, zero_based_numbering=True)
    assert dihedral_idxs ==  [[1,2,3,4]]
    assert dihedral_ranges == []
    # test zero_based_numbering flag in file
    dihedral_str = '''
    # zero_based_numbering
    #i j k l
     1 2 3 4
     2 3 4 5
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    assert dihedral_idxs ==  [[1,2,3,4], [2,3,4,5]]
    assert dihedral_ranges == []
    # test with conflict options
    dihedral_str = '''
    # zero_based_numbering
    # one_based_numbering
    #i j k l
     1 2 3 4
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    with pytest.raises(ValueError):
        dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    # test with conflict options
    dihedral_str = '''
    # one_based_numbering
    # zero_based_numbering
    #i j k l
     1 2 3 4
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    with pytest.raises(ValueError):
        dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    # test with conflict options
    dihedral_str = '''
    # one_based_numbering
    #i j k l
     1 2 3 4
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    with pytest.raises(ValueError):
        dihedral_idxs, dihedral_ranges = load_dihedralfile(fn, zero_based_numbering=True)

def test_load_dihedralfile_limited_ranges(tmpdir):
    tmpdir.chdir()
    fn = 'dihedrals.txt'
    # test basic loading with limited dihedral ranges
    dihedral_str = '''
    #i j k l range_low range_hi
     1 2 3 4   -120      120
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    assert dihedral_idxs ==  [[0,1,2,3]]
    assert dihedral_ranges == [[-120, 120]]
    # test loading 2 dihedrals with limited dihedral ranges
    dihedral_str = '''
    #i j k l range_low range_hi
     1 2 3 4   -120      120
     2 3 4 5   -90       180
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    assert dihedral_idxs ==  [[0,1,2,3], [1,2,3,4]]
    assert dihedral_ranges == [[-120, 120], [-90, 180]]
    # test loading dihedrals with split dihedral ranges
    dihedral_str = '''
    #i j k l range_low range_hi
     1 2 3 4   -120      120
     2 3 4 5    120      240
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    assert dihedral_idxs ==  [[0,1,2,3], [1,2,3,4]]
    assert dihedral_ranges == [[-120, 120], [120, 240]]
    # test loading dihedrals with default ranges
    dihedral_str = '''
    #i j k l range_lo range_hi
     1 2 3 4   -120      120
     2 3 4 5
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    assert dihedral_idxs ==  [[0,1,2,3], [1,2,3,4]]
    assert dihedral_ranges == [[-120, 120], [-180, 180]]
    # test loading dihedrals with default ranges
    dihedral_str = '''
    #i j k l range_lo range_hi
     1 2 3 4   -120      120
     2 3 4 5
     3 4 5 6   -90       150
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    assert dihedral_idxs ==  [[0,1,2,3], [1,2,3,4], [2,3,4,5]]
    assert dihedral_ranges == [[-120, 120], [-180, 180], [-90, 150]]
    # test loading dihedrals with wrong ranges low>=high
    dihedral_str = '''
    #i j k l range_lo range_hi
     1 2 3 4   -120      120
     2 3 4 5    120      120
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    with pytest.raises(AssertionError, match='range'):
        dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    # test loading dihedrals with wrong ranges low<-180
    dihedral_str = '''
    #i j k l range_lo range_hi
     1 2 3 4   -200      120
     2 3 4 5    120      240
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    with pytest.raises(AssertionError, match='range'):
        dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)
    # test loading dihedrals with wrong ranges high>360
    dihedral_str = '''
    #i j k l range_lo range_hi
     1 2 3 4   -120      120
     2 3 4 5    120      361
    '''
    with open(fn, 'w') as fp:
        fp.write(dihedral_str)
    with pytest.raises(AssertionError, match='range'):
        dihedral_idxs, dihedral_ranges = load_dihedralfile(fn)

def test_create_engine(tmpdir):
    """Test making an OpenMM engine with native optimizer, expect AssertionError"""

    with pytest.raises(AssertionError):
        engine = create_engine('openmm', native_opt=True)


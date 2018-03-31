"""
Unit and regression test for the crank package.
"""

# Import package, test suite, and other packages as needed
import crank
import pytest
import sys

def test_crank_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "crank" in sys.modules

def test_import():
    """
    Testing import of crank modules
    """
    from crank import DihedralScanner, QMEngine, PriorityQueue

def test_dihedral_scanner():
    """
    Testing import of crank modules
    """
    from crank import DihedralScanner
    DihedralScanner.test()


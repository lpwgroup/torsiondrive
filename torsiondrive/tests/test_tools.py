"""
Unit and regression test for the torsiondrive.tools package
"""

import os
import sys
import urllib.request
import pytest

def test_torsiondrive_tools():
    """
    Test the torsiondrive.tools package
    """
    import torsiondrive.tools

def test_tools_read_scan_xyz(tmpdir):
    """
    Test torsiondrive.tools.read_scan_xyz
    """
    from torsiondrive.tools import read_scan_xyz
    tmpdir.chdir()
    # Test reading 1D scan.xyz
    # Use an example file downloaded from the torsiondrive_examples repo
    url = "https://raw.githubusercontent.com/lpwgroup/torsiondrive_examples/master/examples/hooh-1d/psi4/run_local/geomeTRIC/scan.xyz"
    urllib.request.urlretrieve(url, 'scan.xyz')
    grid_data = read_scan_xyz('scan.xyz')
    assert len(grid_data) == 24
    assert sorted(grid_data.keys()) == [(gid,) for gid in range(-165, 195, 15)]
    # Test reading 2D scan.xyz
    url = "https://github.com/lpwgroup/torsiondrive_examples/raw/master/examples/range_limited_split/scan.xyz"
    urllib.request.urlretrieve(url, 'scan.xyz')
    grid_data = read_scan_xyz('scan.xyz')
    gid_list = sorted(grid_data.keys())
    assert len(gid_list[0]) == 2
    assert len(gid_list) == 63
    # Test reading an incorrect scan.xyz
    with open('wrong.xyz', 'w') as outfile:
        outfile.write('3\nregular xyz without grid information\nH 0 0 0\nH 0 0 2\nO 0 0 1')
    with pytest.raises(ValueError):
        read_scan_xyz('wrong.xyz')

def test_tools_find_grid_spacing():
    """
    Test torsiondrive.tools.find_grid_spacing
    """
    from torsiondrive.tools import find_grid_spacing
    # regular gid list
    gid_list = list(range(-165, 195, 15))
    assert find_grid_spacing(gid_list) == 15
    gid_list = list(range(-150, 210, 30))
    assert find_grid_spacing(gid_list) == 30
    # gid list with missing values
    gid_list = [-150, -120, -60, -45, 0, 60, 120, 150, 180]
    assert find_grid_spacing(gid_list) == 15
    # special gid list
    gid_list = [-150, -90, -30, 30, 90, 150]
    # -150 distance to -180 is 30
    assert find_grid_spacing(gid_list) == 30
    gid_list = [10, 30, 50, 70]
    # spacing of 20 does not hit -180 and 360, so 10 is largest possible
    assert find_grid_spacing(gid_list) == 10
    gid_list = [70]
    # largest possible grid spacing is 10
    assert find_grid_spacing(gid_list) == 10
    # empty gid list
    assert find_grid_spacing([]) == None

def test_plot_1d_curve(tmpdir):
    """ Test torsiondrive.tools.plot_1d.plot_1d_curve """
    from torsiondrive.tools import read_scan_xyz
    from torsiondrive.tools import plot_1d
    tmpdir.chdir()
    # Plotting empty data should do nothing
    plot_1d.plot_1d_curve({}, 'nothing.pdf')
    assert not os.path.isfile('nothing.pdf')
    # Use an example file downloaded from the torsiondrive_examples repo
    url = "https://raw.githubusercontent.com/lpwgroup/torsiondrive_examples/master/examples/hooh-1d/psi4/run_local/geomeTRIC/scan.xyz"
    urllib.request.urlretrieve(url, 'scan.xyz')
    grid_data = read_scan_xyz('scan.xyz')
    plot_1d.plot_1d_curve(grid_data, '1d.pdf')
    assert os.path.isfile('1d.pdf')
    # test running the main function
    argv = sys.argv[:]
    sys.argv = "torsiondrive-plot1d.py scan.xyz".split()
    plot_1d.main()
    sys.argv = argv
    assert os.path.isfile('scan.pdf')

def test_format_2d_data():
    from torsiondrive.tools.plot_2d import format_2d_grid_data
    import numpy as np
    # Test empty grid data
    grid_data = {}
    with pytest.raises(ValueError):
        format_2d_grid_data(grid_data)
    # Test formatting 1-D grid data, should raise error
    grid_data = {(-165,): -0.1, (-150,): -0.2, (-90,): -0.3}
    with pytest.raises(AssertionError):
        format_2d_grid_data(grid_data)
    # Test formatting 2-D grid data
    grid_x = [-90, 0, 90, 180]
    grid_y = [-60, 60, 180]
    grid_data = {(x,y): -100.0 for x in grid_x for y in grid_y}
    x_array, y_array, z_mat = format_2d_grid_data(grid_data)
    ref_z_mat = np.ones((4, 3), dtype=float) * -100.0
    assert np.array_equal(x_array, grid_x)
    assert np.array_equal(y_array, grid_y)
    assert np.array_equal(z_mat, ref_z_mat)
    # Test formatting and incomplete 2-D grid data
    grid_x = [-90, 90, 180]
    grid_y = [-60, 60, 180]
    grid_data = {(x,y): -100.0 for x in grid_x for y in grid_y}
    x_array, y_array, z_mat = format_2d_grid_data(grid_data)
    ref_x_array = [-90, 0, 90, 180]
    ref_z_mat = np.ones((4, 3), dtype=float) * -100.0
    # missing values should be filled by np.nan
    ref_z_mat[1, :] = np.nan
    assert np.array_equal(x_array, ref_x_array)
    assert np.array_equal(y_array, grid_y)
    assert np.allclose(z_mat, ref_z_mat, equal_nan=True)


def test_plot_2d_contour(tmpdir):
    """ Test torsiondrive.tools.plot_1d.plot_1d_curve """
    from torsiondrive.tools import read_scan_xyz
    from torsiondrive.tools import plot_2d
    tmpdir.chdir()
    # Test reading 1D scan.xyz
    # Use an example file downloaded from the torsiondrive_examples repo
    url = "https://github.com/lpwgroup/torsiondrive_examples/raw/master/examples/range_limited_split/scan.xyz"
    urllib.request.urlretrieve(url, 'scan.xyz')
    grid_data = read_scan_xyz('scan.xyz')
    plot_2d.plot_grid_contour(grid_data, '2d-contour.pdf', method='contourf')
    assert os.path.isfile('2d-contour.pdf')
    plot_2d.plot_grid_contour(grid_data, '2d-heatmap.pdf', method='imshow')
    assert os.path.isfile('2d-heatmap.pdf')
    # test running the main function
    argv = sys.argv[:]
    sys.argv = "torsiondrive-plot2d.py scan.xyz -m contourf -o 2d-plot.pdf".split()
    plot_2d.main()
    sys.argv = argv
    assert os.path.isfile('2d-plot.pdf')
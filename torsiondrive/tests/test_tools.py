"""
Unit and regression test for the torsiondrive.tools package
"""

import os
import sys
import urllib.request

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
    # empty gid list
    assert find_grid_spacing([]) == None

def test_plot_1d_curve(tmpdir):
    """ Test torsiondrive.tools.plot_1d.plot_1d_curve """
    from torsiondrive.tools import read_scan_xyz
    from torsiondrive.tools import plot_1d
    tmpdir.chdir()
    # Test reading 1D scan.xyz
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
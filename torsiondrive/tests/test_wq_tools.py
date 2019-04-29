"""
Test for the wq_tools module.
"""

import pytest
import os
import sys
import subprocess

try:
    import work_queue
except:
    pass

@pytest.mark.skipif("work_queue" not in sys.modules, reason='work_queue not found')
def test_work_queue():
    from torsiondrive.wq_tools import WorkQueue
    wq = WorkQueue(56789)
    wq.submit('echo test > test.txt', [], ['test.txt'])
    assert wq.get_queue_status() == (0,0,0,1)
    # submit a worker
    p = subprocess.Popen("$HOME/opt/cctools/bin/work_queue_worker localhost 56789 -t 1", shell=True, stderr=subprocess.DEVNULL)
    for _ in range(10):
        path = wq.check_finished_task_path()
        if path is not None:
            assert path == os.getcwd()
            break
    wq.print_queue_status()
    p.terminate()
    assert os.path.isfile('test.txt')
    assert open('test.txt').read().strip() == 'test'
    os.unlink('test.txt')

"""
Unit and regression test for the torsiondrive.priority_queue module
"""

import pytest
from torsiondrive.priority_queue import PriorityQueue

def test_priority_queue():
    """ Test torsiondrive.priority_queue.PriorityQueue """
    pq = PriorityQueue()
    # basic push and pop
    task = 'task'
    pq.push(task)
    assert pq.pop() == task
    # push and pop with priority
    task1 = 'task_fjei'
    task2 = 'task_ioue'
    task3 = 'task_asdb'
    pq.push(task1, priority=2)
    pq.push(task2, priority=1)
    pq.push(task3, priority=2)
    # task2 has a smaller priority value -> higher priority
    assert pq.pop() == task2
    # the next one should be task1 because it's pushed earlier
    assert pq.pop() == task1
    # last one is task3
    assert pq.pop() == task3
    # test popping empty error
    with pytest.raises(IndexError):
        pq.pop()
    # test push many
    pq = PriorityQueue()
    tasks = [f'task{i}' for i in range(20)]
    pq.push_many(tasks)
    # test len(pq)
    assert len(pq) == 20
    # test looping over tasks
    for task, task_ref in zip(pq, tasks):
        assert task == task_ref

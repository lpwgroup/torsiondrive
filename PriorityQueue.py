#!/usr/bin/env python

########################
#  PriorityQueue Tool  #
#  Author: Yudong Qiu  #
########################

from heapq import heappush, heappop
import itertools

class PriorityQueue(object):
    def __len__(self):
        return len(self._pq)

    def __init__(self):
        self._pq = []
        self.counter = itertools.count()

    def push(self, task, priority=0):
        """
        Add one task to priority queue
        When priority is the same, count ensures the earlier added tasks first
        """
        count = next(self.counter)
        # -priority used here so the task with higher priority first
        entry = [-priority, count, task]
        heappush(self._pq, entry)

    def pop(self):
        """
        Pop the task that was pushed with highest priority value
        """
        if len(self._pq) == 0:
            raise RuntimeError("pop from an empty priority queue")
        priority, count, task = heappop(self._pq)
        return task

    def push_many(self, tasks, priority=0):
        for task in tasks:
            self.push(task, priority)

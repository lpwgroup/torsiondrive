#!/usr/bin/env python

import itertools
from heapq import heappush, heappop


class PriorityQueue(object):

    def __len__(self):
        return len(self._pq)

    def __iter__(self):
        return iter(task for priority, count, task in self._pq)

    def __init__(self):
        self._pq = []
        self.counter = itertools.count()

    def push(self, task, priority=0):
        """
        Add one task to priority queue
        When priority is the same, count ensures the earlier added tasks first
        """
        count = next(self.counter)
        entry = [priority, count, task]
        heappush(self._pq, entry)

    def pop(self):
        """
        Pop the task that was pushed with highest priority value
        """
        if len(self._pq) == 0:
            raise RuntimeError("pop from an empty priority queue")
        _, _, task = heappop(self._pq)
        return task

    def push_many(self, tasks, priority=0):
        for task in tasks:
            self.push(task, priority)

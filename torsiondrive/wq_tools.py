from __future__ import print_function

import os
import sys
import time

import work_queue


class WorkQueue:
    def __init__(self, port, name='dihedral'):
        work_queue.set_debug_flag('all')
        wq = work_queue.WorkQueue(port=port, exclusive=False, shutdown=False)
        wq.specify_keepalive_interval(8640000)
        wq.specify_name(name)
        self.wq = wq
        self.tasks_failed = 0 # Our own counter for tasks that failed
        self.queue_status = None
        self.last_print_time = 0
        print('Work Queue listening on %d' % wq.port, file=sys.stderr)

    def submit(self, command, inputfiles, outputfiles):
        command += ' 2>&1'
        task = work_queue.Task(command)
        cwd = os.getcwd()
        for f in inputfiles:
            lf = os.path.join(cwd,f)
            task.specify_input_file(lf, f, cache=False)
        for f in outputfiles:
            lf = os.path.join(cwd,f)
            task.specify_output_file(lf, f, cache=False)
        task.specify_algorithm(work_queue.WORK_QUEUE_SCHEDULE_RAND)
        task.specify_tag(cwd)
        task.print_time = 60
        taskid = self.wq.submit(task)
        return taskid

    def check_finished_task_path(self, wait_time=1):
        task = self.wq.wait(wait_time)
        # try to print the status
        self.print_queue_status()
        # if some task finished
        if task:
            exectime = task.cmd_execution_time/1000000
            if task.result == 0:
                # return taskpath if it finished successfully
                print("Command '%s' (task %d) finished successfully on host %s (%d seconds)" % (task.command, task.id, task.hostname, exectime), file=sys.stderr)
                return task.tag
            else:
                # resubmit the task if it failed
                oldid = task.id
                oldhost = task.hostname
                new_taskid = self.wq.submit(task)
                self.tasks_failed += 1
                print("Command '%s' (task %d) failed on host %s (%d seconds), resubmitted: taskid %d" % (task.command, oldid, oldhost, exectime, new_taskid), file=sys.stderr)
        else:
            return None

    def get_queue_status(self):
        """
        Get the current queue status.
        Return a tuple (n_running_workers, n_all_workers, n_finished_jobs, n_total_jobs)
        """
        stats = self.wq.stats
        n_running_workers = stats.workers_busy
        n_all_workers = stats.total_workers_joined - stats.total_workers_removed
        n_finished_jobs = stats.total_tasks_complete - self.tasks_failed
        n_total_jobs = stats.total_tasks_dispatched - self.tasks_failed
        return n_running_workers, n_all_workers, n_finished_jobs, n_total_jobs

    def print_queue_status(self, min_time_interval=10, max_time_interval=3600):
        """
        Print the current status of the work queue to sys.stderr. Example:
        'Sat Mar 10 10:19:35 2018: 5/10 workers busy; 20/30 jobs complete'
        The print will happen under either condition of the following two:
        1. Time has passed over the max_time_interval since the last print.
        2. Time has passed over the min_time_interval since the last print, and the queue status has changed.
        """
        current_time = time.time()
        time_passed = current_time - self.last_print_time
        # return if we just printed
        if time_passed < min_time_interval: return
        current_status = self.get_queue_status()
        if (time_passed > max_time_interval) or (current_status != self.queue_status):
            status_str = time.ctime()
            status_str += ': %d/%d workers busy; %d/%d jobs complete' % current_status
            print(status_str, file=sys.stderr)
            # update the status and last print time
            self.queue_status, self.last_print_time = current_status, current_time

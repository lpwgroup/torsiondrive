import os
import work_queue

class WorkQueue:
    def __init__(self, port, name='dihedral'):
        #work_queue.set_debug_flag('all')
        wq = work_queue.WorkQueue(port=port, exclusive=False, shutdown=False)
        #wq.tasks_failed = 0 # Counter for tasks that fail at the application level
        wq.specify_keepalive_interval(8640000)
        wq.specify_name(name)
        print('Work Queue listening on %d' % (wq.port))
        self.wq = wq

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
        # if some task finished
        if task:
            exectime = task.cmd_execution_time/1000000
            if task.result == 0:
                # return taskpath if it finished successfully
                print("Command '%s' (task %i) finished successfully on host %s (%i seconds)" % (task.command, task.id, task.hostname, exectime))
                return task.tag
            else:
                # resubmit the task if it failed
                oldid = task.id
                oldhost = task.hostname
                new_taskid = self.wq.submit(task)
                print(("Command '%s' (task %i) failed on host %s (%i seconds), resubmitted: taskid %i" % (task.command, oldid, oldhost, exectime, new_taskid)))
        else:
            return None

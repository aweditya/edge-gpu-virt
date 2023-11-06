#ifndef _FCFS_SCHEDULER_H
#define _FCFS_SCHEDULER_H

#include <pthread.h>
#include <queue>

#include "Scheduler.h"

class FCFSScheduler : public Scheduler
{
public:
    FCFSScheduler() { pthread_mutex_init(&mutex, NULL); }
    ~FCFSScheduler() { pthread_mutex_destroy(&mutex); }

    void scheduleKernel(kernel_attr_t *kernel);

private:
    std::queue<kernel_attr_t *> activeKernels;
    pthread_mutex_t mutex;

    void schedule();
};
#endif
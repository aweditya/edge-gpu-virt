#ifndef _PRIORITY_SCHEDULER_H
#define _PRIORITY_SCHEDULER_H

#include <pthread.h>
#include <queue>

#include "Scheduler.h"

class PriorityScheduler : public Scheduler
{
public:
    PriorityScheduler() { pthread_mutex_init(&mutex, NULL); }
    ~PriorityScheduler() { pthread_mutex_destroy(&mutex); }

    void scheduleKernel(kernel_attr_t *kernel);

private:
    std::priority_queue<kernel_attr_t *, std::vector<kernel_attr_t *>, compare_kernel_attr> activeKernels;
    pthread_mutex_t mutex;

    void schedule();
};
#endif
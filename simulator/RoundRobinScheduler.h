#ifndef _ROUND_ROBIN_SCHEDULER_H
#define _ROUND_ROBIN_SCHEDULER_H

#include <pthread.h>
#include <vector>

#include "Scheduler.h"

class RoundRobinScheduler : public Scheduler
{
public:
    RoundRobinScheduler() { pthread_mutex_init(&mutex, NULL); }
    ~RoundRobinScheduler() { pthread_mutex_destroy(&mutex); }

    void scheduleKernel(kernel_attr_t *kernel);

private:
    std::vector<kernel_attr_t *> activeKernels;
    pthread_mutex_t mutex;

    void schedule();
};
#endif
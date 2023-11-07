#include "PriorityScheduler.h"

void PriorityScheduler::scheduleKernel(kernel_attr_t *kernel)
{
    kernel->id = rand();
    set_state(&(kernel->kcb), LAUNCH);

    pthread_mutex_lock(&mutex);
    activeKernels.push(kernel);
    pthread_mutex_unlock(&mutex);
}

void PriorityScheduler::schedule()
{
    if (activeKernels.size() == 0)
    {
        usleep(1);
    }
    else
    {
        pthread_mutex_lock(&mutex);
        activeKernels.top()->kcb.slicesToLaunch = 2;
        launchKernel(activeKernels.top());
        if (activeKernels.top()->kcb.totalSlices == 0)
        {
            set_state(&(activeKernels.top()->kcb), MEMCPYDTOH, true);

            activeKernels.pop();
        }
        pthread_mutex_unlock(&mutex);
    }
}
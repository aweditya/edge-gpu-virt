#include "FCFSScheduler.h"

void FCFSScheduler::scheduleKernel(kernel_attr_t *kernel)
{
    kernel->id = rand();
    set_state(&(kernel->kcb), LAUNCH);

    pthread_mutex_lock(&mutex);
    activeKernels.push(kernel);
    pthread_mutex_unlock(&mutex);
}

void FCFSScheduler::schedule()
{
    if (activeKernels.size() == 0)
    {
        usleep(1);
    }
    else
    {
        printf("[thread id: %ld] number of kernels: %ld\n", pthread_self(), activeKernels.size());
        activeKernels.front()->kcb.slicesToLaunch = 2;
        launchKernel(activeKernels.front());
        if (activeKernels.front()->kcb.totalSlices == 0)
        {
            set_state(&(activeKernels.front()->kcb), MEMCPYDTOH, true);
            pthread_mutex_lock(&mutex);
            activeKernels.pop();
            pthread_mutex_unlock(&mutex);
        }
    }
}
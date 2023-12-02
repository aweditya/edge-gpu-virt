#include "RoundRobinScheduler.h"

void RoundRobinScheduler::scheduleKernel(kernel_attr_t *kernel)
{
    kernel->id = rand();
    set_state(&(kernel->kcb), LAUNCH);

    pthread_mutex_lock(&mutex);
    activeKernels.push_back(kernel);
    pthread_mutex_unlock(&mutex);
}

void RoundRobinScheduler::schedule()
{
    if (activeKernels.size() == 0)
    {
        usleep(1);
    }
    else
    {
        pthread_mutex_lock(&mutex);
        for (auto it = activeKernels.begin(); it != activeKernels.end();)
        {
            (*it)->kcb.slicesToLaunch = 2;
            launchKernel(*it);

            if ((*it)->kcb.totalSlices == 0)
            {
                set_state(&((*it)->kcb), MEMCPYDTOH, true);
                it = activeKernels.erase(it);
            }
            else
            {
                ++it;
            }
        }
        pthread_mutex_unlock(&mutex);
    }
}
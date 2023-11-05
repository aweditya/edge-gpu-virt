#ifndef _SCHEDULER_H
#define _SCHEDULER_H

#include <vector>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "KernelControlBlock.h"
#include "KernelAttributes.h"
#include "errchk.h"

class Scheduler
{
public:
    Scheduler() {}
    ~Scheduler() {}

    void scheduleKernel(kernel_attr_t *kernel);
    void launchKernel(kernel_attr_t *kernel);

    void run()
    {
        pthread_create(&schedulerThread, NULL, threadFunction, this);
    }

    void finish()
    {
        pthread_join(schedulerThread, NULL);
    }

private:
    pthread_t schedulerThread;
    std::vector<kernel_attr_t *> activeKernels;

    void *threadFunction()
    {
        while (true)
        {
            if (activeKernels.size() == 0)
            {
                usleep(1);
                continue;
            }
            else
            {
                printf("[thread id: %ld] number of kernels: %ld\n", pthread_self(), activeKernels.size());
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
            }
        }
    }

    static void *threadFunction(void *args)
    {
        Scheduler *scheduler = static_cast<Scheduler *>(args);
        scheduler->threadFunction();
    }
};

#endif
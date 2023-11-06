#ifndef _SCHEDULER_H
#define _SCHEDULER_H

#include <unistd.h>
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
    Scheduler(bool *done) : done(done) { pthread_mutex_init(&mutex, NULL); }
    ~Scheduler() { pthread_mutex_destroy(&mutex); }

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
    bool *done;
    pthread_t schedulerThread;
    pthread_mutex_t mutex;
    std::vector<kernel_attr_t *> activeKernels;

    void *threadFunction()
    {
        while (!(*done))
        {
            if (activeKernels.size() == 0)
            {
                usleep(1);
                continue;
            }
            else
            {
                printf("[thread id: %ld] number of kernels: %ld\n", pthread_self(), activeKernels.size());
                activeKernels[0]->kcb.slicesToLaunch = 2;
                launchKernel(activeKernels[0]);

                if (activeKernels[0]->kcb.totalSlices == 0)
                {
                    set_state(&(activeKernels[0]->kcb), MEMCPYDTOH, true);
                    pthread_mutex_lock(&mutex);
                    activeKernels.erase(activeKernels.begin());
                    pthread_mutex_unlock(&mutex);
                }

                // for (auto it = activeKernels.begin(); it != activeKernels.end();)
                // {
                //     (*it)->kcb.slicesToLaunch = 2;
                //     launchKernel(*it);

                //     if ((*it)->kcb.totalSlices == 0)
                //     {
                //         set_state(&((*it)->kcb), MEMCPYDTOH, true);
                //         pthread_mutex_lock(&mutex);
                //         it = activeKernels.erase(it);
                //         pthread_mutex_unlock(&mutex);
                //     }
                //     else
                //     {
                //         ++it;
                //     }
                // }
            }
        }
        return nullptr;
    }

    static void *threadFunction(void *args)
    {
        Scheduler *scheduler = static_cast<Scheduler *>(args);
        return scheduler->threadFunction();
    }
};

#endif
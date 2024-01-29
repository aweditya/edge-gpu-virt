#ifndef _SM_THREAD_OCCUPANCY_PROFILER_H
#define _SM_THREAD_OCCUPANCY_PROFILER_H

#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "errchk.h"

class SMThreadOccupancyProfiler
{
public:
    SMThreadOccupancyProfiler(int multiprocessorCount, int loggingInterval, int loggingDuration) : multiprocessorCount(multiprocessorCount),
                                                                                                   loggingInterval(loggingInterval),
                                                                                                   loggingDuration(loggingDuration) {}

    ~SMThreadOccupancyProfiler() {}

    void launch()
    {
        checkCudaErrors(cuMemAllocManaged(&perSMThreads, sizeof(int) * multiprocessorCount, CU_MEM_ATTACH_GLOBAL));
        perSMThreads_host = (int *)perSMThreads;

        for (int i = 0; i < multiprocessorCount; ++i)
        {
            perSMThreads_host[i] = 0;
        }
        pthread_create(&thread, NULL, threadFunction, this);
    }

    void finish()
    {
        pthread_join(thread, NULL);
        checkCudaErrors(cuMemFree(perSMThreads));
        perSMThreads_host = NULL;
    }

    int *perSMThreads_host;
    CUdeviceptr perSMThreads;

private:
    int multiprocessorCount; // Number of SMs on GPU (retrieved from device properties)
    int loggingInterval;     // Logging interval (in us)
    int loggingDuration;     // Logging duration (in ms)
    pthread_t thread;

    void reportPerSMThreads()
    {
        printf("[thread id: %ld] ", pthread_self());
        for (int i = 0; i < multiprocessorCount; ++i)
        {
            printf("%d\t", perSMThreads_host[i]);
        }
        printf("\n");
    }

    void *threadFunction()
    {
        int iterations = loggingDuration * 1000 / loggingInterval;
        for (int i = 0; i < iterations; ++i)
        {
            reportPerSMThreads();
            usleep(loggingInterval);
        }

        return NULL;
    }

    static void *threadFunction(void *args)
    {
        SMThreadOccupancyProfiler *kernelProfiler = (SMThreadOccupancyProfiler *)args;
        return kernelProfiler->threadFunction();
    }
};

#endif // _SM_THREAD_OCCUPANCY_PROFILER_H
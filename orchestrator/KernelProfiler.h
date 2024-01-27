#ifndef _KERNEL_PROFILER_H
#define _KERNEL_PROFILER_H

#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "errchk.h"

class KernelProfiler
{
public:
    KernelProfiler(int multiprocessorCount, int loggingInterval, int loggingDuration) : multiprocessorCount(multiprocessorCount),
                                                                                                            loggingInterval(loggingInterval),
                                                                                                            loggingDuration(loggingDuration)
    {
        checkCudaErrors(cuMemAllocManaged(&perSMThreads, sizeof(int) * multiprocessorCount, CU_MEM_ATTACH_GLOBAL));
        checkCudaErrors(cuPointerGetAttribute((void *)perSMThreads_host, CU_POINTER_ATTRIBUTE_HOST_POINTER, perSMThreads));

        for (int i = 0; i < multiprocessorCount; ++i)
        {
            perSMThreads_host[i] = 0;
        }
    }

    ~KernelProfiler()
    {
        checkCudaErrors(cuMemFree(perSMThreads));
    }

    void launch()
    {
        pthread_create(&thread, NULL, threadFunction, this);
    }

    void finish()
    {
        pthread_join(thread, NULL);
    }

    int *perSMThreads_host;
    CUdeviceptr perSMThreads;

private:
    int multiprocessorCount; // Number of SMs on GPU (retrieved from device properties)
    int loggingInterval;     // Logging interval (in ms)
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
        printf("void *threadFunction()\n");
        int iterations = loggingDuration / loggingInterval;
        for (int i = 0; i < iterations; ++i)
        {
            printf("loop\n");
            reportPerSMThreads();
            usleep(loggingInterval * 1000);
        }

        return NULL;
    }

    static void *threadFunction(void *args)
    {
        KernelProfiler *kernelProfiler = (KernelProfiler *)args;
        return kernelProfiler->threadFunction();
    }
};

#endif // _KERNEL_PROFILER_H
#pragma once

#include <cuda.h>

#include "kernel.h"
#include "workload_manager.h"

class Client
{
public:
    Client(WorkloadManager *manager, cudaStream_t *offeredStream) : clientManager(manager)
    {
        kernel.clientStream = offeredStream;
    }

    ~Client()
    {
        if (kernel.clientStream)
        {
            cudaStreamDestroy(*(kernel.clientStream));
        }
    }
    static void *threadFunction(void *arg);
    void *threadFunction();

    virtual void setupKernel() = 0;  // Initialize host, device pointers; transfer data from host->device
    virtual void finishKernel() = 0; // Transfer data from device->host; free host, device pointers

protected:
    WorkloadManager *clientManager;
    pthread_t clientThread;

    kernel_args_t kernel;
    int blockOffsetx, blockOffsety; // For kernel slicing
};

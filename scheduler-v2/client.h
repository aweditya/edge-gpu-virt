#pragma once

#include <pthread.h>
#include <cuda.h>

#include "kernel.h"
#include "workload_manager.h"

class Client
{
public:
    Client(WorkloadManager *manager, cudaStream_t *offeredStream) : clientManager(manager), clientStream(offeredStream)
    {
        pthread_create(&clientThread, NULL, &threadFunction, this);
    }
    ~Client()
    {
        pthread_join(clientThread, nullptr);
        if (clientStream)
        {
            cudaStreamDestroy(clientStream);
        }
    }

    virtual void setupKernel() = 0;  // Initialize host, device pointers; transfer data from host->device
    virtual void finishKernel() = 0; // Transfer data from device->host; free host, device pointers

private:
    WorkloadManager *clientManager;
    pthread_t clientThread;
    cudaStream_t *clientStream;
    kernel_args_t kernel;

    static void *threadFunction(void *arg)
    {
        Client *client = static_cast<Client *>(arg);

        while (true)
        {
            setupKernel();

            // Send the CUDA kernel to the main thread for execution
            clientManager->requestLaunchKernel(&kernel);

            // Continue running application code inside thread

            finishKernel();
        }

        return nullptr;
    }
}
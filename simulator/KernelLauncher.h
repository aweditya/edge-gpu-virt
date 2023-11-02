#ifndef _KERNEL_LAUNCHER_H
#define _KERNEL_LAUNCHER_H

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "KernelCallback.h"
#include "errchk.h"

class KernelLauncher
{
public:
    KernelLauncher(const std::string &moduleFile,
                   const std::string &kernelName,
                   unsigned int gridDimX,
                   unsigned int gridDimY,
                   unsigned int gridDimZ,
                   unsigned int blockDimX,
                   unsigned int blockDimY,
                   unsigned int blockDimZ,
                   unsigned int sharedMemBytes,
                   const CUstream &stream,
                   KernelCallback *kernelCallback) : moduleFile(moduleFile),
                                                     kernelName(kernelName),
                                                     gridDimX(gridDimX),
                                                     gridDimY(gridDimY),
                                                     gridDimZ(gridDimZ),
                                                     blockDimX(blockDimX),
                                                     blockDimY(blockDimY),
                                                     blockDimZ(blockDimZ),
                                                     sharedMemBytes(sharedMemBytes),
                                                     stream(stream)
    {
        callback = kernelCallback;
        kernelParams = callback->args;
        callback->setLauncherID(rand());
    }

    ~KernelLauncher()
    {
        pthread_join(thread, NULL);
    }

    void launch()
    {
        pthread_create(&thread, NULL, threadFunction, this);
    }

private:
    pthread_t thread;
    std::string moduleFile;
    std::string kernelName;
    CUmodule module;
    CUfunction function;

    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream stream;
    void **kernelParams;
    KernelCallback *callback;

    static void *threadFunction(void *args)
    {
        KernelLauncher *kernelLauncher = static_cast<KernelLauncher *>(args);
        return kernelLauncher->threadFunction();
    }
    void *threadFunction();

    void launchKernel()
    {
        checkCudaErrors(cuLaunchKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, NULL));
    }
};

#endif

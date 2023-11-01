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
    KernelLauncher(const std::string &moduleFile, const std::string &kernelName, KernelCallback *kernelCallback)
    {
        checkCudaErrors(cuModuleLoad(&module, moduleFile.c_str()));
        checkCudaErrors(cuModuleGetFunction(&function, module, kernelName.c_str()));
        callback = kernelCallback;
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
    CUmodule module;
    CUfunction function;
    KernelCallback *callback;

    static void *threadFunction(void *args)
    {
        KernelLauncher *kernelLauncher = static_cast<KernelLauncher *>(args);
        return kernelLauncher->threadFunction();
    }
    void *threadFunction();
};

#endif

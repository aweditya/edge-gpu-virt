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
                   const CUstream &stream,
                   KernelCallback *kernelCallback) : moduleFile(moduleFile), kernelName(kernelName), stream(stream)
    {
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
    std::string moduleFile;
    std::string kernelName;
    CUmodule module;
    CUfunction function;
    CUstream stream;
    KernelCallback *callback;

    static void *threadFunction(void *args)
    {
        KernelLauncher *kernelLauncher = static_cast<KernelLauncher *>(args);
        return kernelLauncher->threadFunction();
    }
    void *threadFunction();
};

#endif

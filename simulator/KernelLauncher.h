#ifndef _KERNEL_LAUNCHER_H
#define _KERNEL_LAUNCHER_H

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "KernelCallback.h"
#include "errchk.h"

typedef struct kernel_attr
{
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream stream;
} kernel_attr_t;

class KernelLauncher
{
public:
    KernelLauncher(int id,
                   CUcontext *context,
                   const std::string &moduleFile,
                   const std::string &kernelName,
                   kernel_attr_t *attr,
                   KernelCallback *kernelCallback) : id(id),
                                                     context(context),
                                                     moduleFile(moduleFile),
                                                     kernelName(kernelName),
                                                     attr(attr)
    {
        callback = kernelCallback;
        kernelParams = callback->args;
        callback->setLauncherID(id);
    }

    ~KernelLauncher() {}

    void launch()
    {
        pthread_create(&thread, NULL, threadFunction, this);
    }

    void finish()
    {
        pthread_join(thread, NULL);
    }

private:
    int id;
    CUcontext *context;
    pthread_t thread;
    std::string moduleFile;
    std::string kernelName;
    CUmodule module;
    CUfunction function;

    kernel_attr_t *attr;
    void **kernelParams;
    KernelCallback *callback;

    void *threadFunction()
    {
        checkCudaErrors(cuCtxSetCurrent(*context));
        checkCudaErrors(cuModuleLoad(&module, moduleFile.c_str()));
        checkCudaErrors(cuModuleGetFunction(&function, module, kernelName.c_str()));
        callback->memAlloc();
        callback->memcpyHtoD(attr->stream);
        launchKernel();
        callback->memcpyDtoH(attr->stream);
        callback->memFree();

        return NULL;
    }

    static void *threadFunction(void *args)
    {
        KernelLauncher *kernelLauncher = static_cast<KernelLauncher *>(args);
        return kernelLauncher->threadFunction();
    }

    void launchKernel()
    {
        checkCudaErrors(cuLaunchKernel(function,
                                       attr->gridDimX,
                                       attr->gridDimY,
                                       attr->gridDimZ,
                                       attr->blockDimX,
                                       attr->blockDimY,
                                       attr->blockDimZ,
                                       attr->sharedMemBytes,
                                       attr->stream,
                                       kernelParams,
                                       0));
    }
};

#endif

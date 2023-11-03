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

    unsigned int sGridDimX;
    unsigned int sGridDimY;
    unsigned int sGridDimZ;

    unsigned int blockOffsetX = 0;
    unsigned int blockOffsetY = 0;
    unsigned int blockOffsetZ = 0;

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
        
        callback->args[0] = &(attr->blockOffsetX);
        callback->args[1] = &(attr->blockOffsetY);
        callback->args[2] = &(attr->blockOffsetZ);

        totalSlices = (attr->gridDimX * attr->gridDimY * attr->gridDimZ) / (attr->sGridDimX * attr->sGridDimY * attr->sGridDimZ);
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

    unsigned int totalSlices;

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
        while (totalSlices--)
        {
            printf("[%d] slices left = %d\n", id, totalSlices);
            checkCudaErrors(cuLaunchKernel(function,
                                           attr->sGridDimX,
                                           attr->sGridDimY,
                                           attr->sGridDimZ,
                                           attr->blockDimX,
                                           attr->blockDimY,
                                           attr->blockDimZ,
                                           attr->sharedMemBytes,
                                           attr->stream,
                                           kernelParams,
                                           nullptr));

            attr->blockOffsetX += attr->sGridDimX;
            while (attr->blockOffsetX >= attr->gridDimX)
            {
                attr->blockOffsetX -= attr->gridDimX;
                attr->blockOffsetY += attr->sGridDimY;
            }

            while (attr->blockOffsetY >= attr->gridDimY)
            {
                attr->blockOffsetY -= attr->gridDimY;
                attr->blockOffsetZ += attr->sGridDimZ;
            }

        }
    }
};

#endif

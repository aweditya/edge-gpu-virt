#ifndef _KERNEL_LAUNCHER_H
#define _KERNEL_LAUNCHER_H

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "Scheduler.h"
#include "KernelControlBlock.h"
#include "KernelCallback.h"
#include "errchk.h"

class KernelLauncher
{
public:
    KernelLauncher(Scheduler *scheduler,
                   CUcontext &context,
                   const std::string &moduleFile,
                   const std::string &kernelName,
                   kernel_attr_t *attr,
                   KernelCallback *kernelCallback) : scheduler(scheduler),
                                                     context(context),
                                                     moduleFile(moduleFile),
                                                     kernelName(kernelName),
                                                     attr(attr)
    {
        callback = kernelCallback;
        attr->kernelParams = callback->args;

        callback->args[0] = &(attr->blockOffsetX);
        callback->args[1] = &(attr->blockOffsetY);
        callback->args[2] = &(attr->blockOffsetZ);

        kernel_control_block_init(&(attr->kcb), (attr->gridDimX * attr->gridDimY * attr->gridDimZ) / (attr->sGridDimX * attr->sGridDimY * attr->sGridDimZ));
    }

    ~KernelLauncher() {}

    void launch()
    {
        pthread_create(&thread, NULL, threadFunction, this);
    }

    void finish()
    {
        printf("[thread id: %ld] exiting...\n", pthread_self());
        pthread_join(thread, NULL);
        kernel_control_block_destroy(&(attr->kcb));
    }

    kernel_attr_t *getKernelAttributes() { return attr; }

private:
    CUcontext context;
    pthread_t thread;
    std::string moduleFile;
    std::string kernelName;
    CUmodule module;

    kernel_attr_t *attr;

    KernelCallback *callback;
    Scheduler *scheduler;

    void *threadFunction()
    {
        checkCudaErrors(cuCtxSetCurrent(context));
        checkCudaErrors(cuModuleLoad(&module, moduleFile.c_str()));
        checkCudaErrors(cuModuleGetFunction(&(attr->function), module, kernelName.c_str()));

        callback->memAlloc();
        callback->memcpyHtoD(attr->stream);

        pthread_mutex_lock(&(attr->kcb.kernel_lock));
        attr->kcb.state = MEMCPYHTOD;
        pthread_mutex_unlock(&(attr->kcb.kernel_lock));

        scheduler->scheduleKernel(this->attr);

        pthread_mutex_lock(&(attr->kcb.kernel_lock));
        while (attr->kcb.state != MEMCPYDTOH)
        {
            pthread_cond_wait(&(attr->kcb.kernel_signal), &(attr->kcb.kernel_lock));
        }
        pthread_mutex_unlock(&(attr->kcb.kernel_lock));

        callback->memcpyDtoH(attr->stream);
        callback->memFree();

        return NULL;
    }

    static void *threadFunction(void *args)
    {
        KernelLauncher *kernelLauncher = static_cast<KernelLauncher *>(args);
        return kernelLauncher->threadFunction();
    }
};

#endif

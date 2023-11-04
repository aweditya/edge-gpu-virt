#ifndef _KERNEL_LAUNCHER_H
#define _KERNEL_LAUNCHER_H

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "Scheduler.h"
#include "KernelCallback.h"
#include "errchk.h"

typedef struct kernel_attr
{
    CUfunction function;

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
    void **kernelParams;
} kernel_attr_t;

enum kstate
{
    INIT = 0,
    MEMCPYHTOD = 1,
    LAUNCH = 2,
    MEMCPYDTOH = 3
};

typedef struct kernel_control_block
{
    pthread_mutex_t kernel_lock;
    pthread_cond_t kernel_signal;
    kstate state;
    unsigned int slicesToLaunch;
    unsigned int totalSlices;
} kernel_control_block_t;

void kernel_control_block_init(kernel_control_block_t *kcb, unsigned int totalSlices)
{
    pthread_mutex_init(&(kcb->kernel_lock), NULL);
    pthread_cond_init(&(kcb->kernel_signal), NULL);
    kcb->state = INIT;
    kcb->slicesToLaunch = 1;
    kcb->totalSlices = totalSlices;
}

void kernel_control_block_destroy(kernel_control_block *kcb)
{
    pthread_mutex_destroy(&(kcb->kernel_lock));
    pthread_cond_destroy(&(kcb->kernel_signal));
}

class KernelLauncher
{
public:
    KernelLauncher(Scheduler *scheduler,
                   int id,
                   CUcontext &context,
                   const std::string &moduleFile,
                   const std::string &kernelName,
                   kernel_attr_t *attr,
                   KernelCallback *kernelCallback) : scheduler(scheduler),
                                                     id(id),
                                                     context(context),
                                                     moduleFile(moduleFile),
                                                     kernelName(kernelName),
                                                     attr(attr)
    {
        callback = kernelCallback;
        attr->kernelParams = callback->args;
        callback->setLauncherID(id);

        callback->args[0] = &(attr->blockOffsetX);
        callback->args[1] = &(attr->blockOffsetY);
        callback->args[2] = &(attr->blockOffsetZ);

        kernel_control_block_init(kcb, (attr->gridDimX * attr->gridDimY * attr->gridDimZ) / (attr->sGridDimX * attr->sGridDimY * attr->sGridDimZ));
    }

    ~KernelLauncher() {}

    void launch()
    {
        pthread_create(&thread, NULL, threadFunction, this);
    }

    void finish()
    {
        pthread_join(thread, NULL);
        kernel_control_block_destroy(kcb);
    }

    int getId() { return id; }
    kernel_attr_t *getKernelAttributes() { return attr; }
    kernel_control_block_t *getKernelControlBlock() { return kcb; }

private:
    int id;
    CUcontext context;
    pthread_t thread;
    std::string moduleFile;
    std::string kernelName;
    CUmodule module;

    kernel_attr_t *attr;
    kernel_control_block_t *kcb;

    KernelCallback *callback;
    Scheduler *scheduler;

    void *threadFunction()
    {
        checkCudaErrors(cuCtxSetCurrent(context));
        checkCudaErrors(cuModuleLoad(&module, moduleFile.c_str()));
        checkCudaErrors(cuModuleGetFunction(&(attr->function), module, kernelName.c_str()));

        callback->memAlloc();
        callback->memcpyHtoD(attr->stream);

        pthread_mutex_lock(&(kcb->kernel_lock));
        kcb->state = MEMCPYHTOD;
        pthread_mutex_unlock(&(kcb->kernel_lock));

        scheduler->scheduleKernel(this);

        pthread_mutex_lock(&kcb->kernel_lock);
        while (kcb->state != MEMCPYDTOH)
        {
            pthread_cond_wait(&kcb->kernel_signal, &kcb->kernel_lock);
        }
        pthread_mutex_unlock(&kcb->kernel_lock);

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

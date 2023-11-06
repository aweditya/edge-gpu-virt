#ifndef _SCHEDULER_H
#define _SCHEDULER_H

#include <unistd.h>
#include <vector>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "KernelControlBlock.h"
#include "KernelAttributes.h"
#include "errchk.h"

class Scheduler
{
public:
    Scheduler() : done(false) {}
    ~Scheduler() {}

    virtual void scheduleKernel(kernel_attr_t *kernel) = 0;
    void launchKernel(kernel_attr_t *kernel)
    {
        for (int i = 0; i < min(kernel->kcb.slicesToLaunch, kernel->kcb.totalSlices); ++i)
        {
            printf("[kernel id: %d] slices left = %d\n", kernel->id, kernel->kcb.totalSlices);
            checkCudaErrors(cuLaunchKernel(kernel->function,
                                           kernel->sGridDimX,
                                           kernel->sGridDimY,
                                           kernel->sGridDimZ,
                                           kernel->blockDimX,
                                           kernel->blockDimY,
                                           kernel->blockDimZ,
                                           kernel->sharedMemBytes,
                                           kernel->stream,
                                           kernel->kernelParams,
                                           nullptr));

            kernel->blockOffsetX += kernel->sGridDimX;
            while (kernel->blockOffsetX >= kernel->gridDimX)
            {
                kernel->blockOffsetX -= kernel->gridDimX;
                kernel->blockOffsetY += kernel->sGridDimY;
            }

            while (kernel->blockOffsetY >= kernel->gridDimY)
            {
                kernel->blockOffsetY -= kernel->gridDimY;
                kernel->blockOffsetZ += kernel->sGridDimZ;
            }

            --kernel->kcb.totalSlices;
        }
    }

    void run()
    {
        pthread_create(&schedulerThread, NULL, threadFunction, this);
    }

    void stop()
    {
        done = true;
    }

    void finish()
    {
        pthread_join(schedulerThread, NULL);
    }

protected:
    bool done;
    pthread_t schedulerThread;

    virtual void schedule() = 0;

    void *threadFunction()
    {
        while (!done)
        {
            schedule();
        }
        return nullptr;
    }

    static void *threadFunction(void *args)
    {
        Scheduler *scheduler = static_cast<Scheduler *>(args);
        return scheduler->threadFunction();
    }
};

#endif
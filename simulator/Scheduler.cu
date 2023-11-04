#include "Scheduler.h"

void Scheduler::scheduleKernel(kernel_attr_t *kernel)
{
    kernel->id = rand();
    kernel->kcb.state = LAUNCH;
    activeKernels.push_back(kernel);
}

void Scheduler::launchKernel(kernel_attr_t *kernel)
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
    }

    kernel->kcb.totalSlices -= kernel->kcb.slicesToLaunch;
}
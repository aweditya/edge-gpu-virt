#include "Scheduler.h"

void Scheduler::scheduleKernel(KernelLauncher *kernel)
{
    kernel_control_block_t *kcb = kernel->getKernelControlBlock();
    kcb->state = LAUNCH;
    activeKernels.push_back(kernel);
}

void Scheduler::launchKernel(KernelLauncher *kernel)
{
    int id = kernel->getId();
    kernel_attr_t *attr = kernel->getKernelAttributes();
    kernel_control_block_t *kcb = kernel->getKernelControlBlock();
    for (int i = 0; i < min(kcb->slicesToLaunch, kcb->totalSlices); ++i)
    {
        printf("[%d] slices left = %d\n", id, kcb->totalSlices);
        checkCudaErrors(cuLaunchKernel(attr->function,
                                       attr->sGridDimX,
                                       attr->sGridDimY,
                                       attr->sGridDimZ,
                                       attr->blockDimX,
                                       attr->blockDimY,
                                       attr->blockDimZ,
                                       attr->sharedMemBytes,
                                       attr->stream,
                                       attr->kernelParams,
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

    kcb->totalSlices -= kcb->slicesToLaunch;
}
#include "workload_manager.h"
#include <stdio.h>

void WorkloadManager::requestLaunchKernel(kernel_args_t *kernel)
{
    kernel_metadata_t newKernel;
    newKernel.kernel = kernel;
    newKernel.slicedGridConf = kernel->gridConf;
    newKernel.totalSlices = 1;
    newKernel.slicesToLaunch = 1;
    activeKernels.emplace_back(newKernel);
}

void WorkloadManager::run()
{
    int launch = 0;
    printf("ola\n");
    while (!activeKernels.empty())
    {
        int kernelToLaunch = launch;

        // Launch the selected kernel asynchronously on the main thread
        cudaLaunchKernel(activeKernels[kernelToLaunch].kernel->kernelFunction,
                         activeKernels[kernelToLaunch].slicedGridConf,
                         activeKernels[kernelToLaunch].kernel->blockConf,
                         activeKernels[kernelToLaunch].kernel->arguments,
                         activeKernels[kernelToLaunch].kernel->sharedMem,
                         *(activeKernels[kernelToLaunch].kernel->clientStream));
        
        activeKernels[kernelToLaunch].totalSlices--;

        // Remove the kernel if all slices have been launched
        if (activeKernels[kernelToLaunch].totalSlices == 0)
        {
            activeKernels.erase(activeKernels.begin() + kernelToLaunch);
        }

    }
}
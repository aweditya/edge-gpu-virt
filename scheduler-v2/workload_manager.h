#include <vector>
#include <cuda.h>

#include "kernel.h"

class WorkloadManager
{
    WorkloadManager();
    ~WorkloadManager();

    void requestLaunchKernel(kernel_args_t *kernel);
    void run();

private:
    std::vector<kernel_metadata_t> activeKernels;
};
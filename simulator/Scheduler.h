#ifndef _SCHEDULER_H
#define _SCHEDULER_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "KernelControlBlock.h"
#include "KernelAttributes.h"
#include "errchk.h"

class Scheduler
{
public:
    Scheduler() {}
    ~Scheduler() {}

    void scheduleKernel(kernel_attr_t *kernel);
    void launchKernel(kernel_attr_t *kernel);

    std::vector<kernel_attr_t *> activeKernels;
};

#endif
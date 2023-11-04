#ifndef _SCHEDULER_H
#define _SCHEDULER_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "KernelLauncher.h"
#include "errchk.h"

class Scheduler
{
public:
    Scheduler() {}
    ~Scheduler() {}

    void scheduleKernel(KernelLauncher *kernel);
    void launchKernel(KernelLauncher *kernel);

    std::vector<KernelLauncher *> activeKernels;
};

#endif
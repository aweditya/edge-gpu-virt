#ifndef _CLOCK_BLOCK_KERNEL_H
#define _CLOCK_BLOCK_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "clockBlock.h"
#include "Kernel.h"

class ClockBlockKernel : public Kernel
{
public:
    ClockBlockKernel(int clockRate) : clockRate(clockRate) {}
    ~ClockBlockKernel() {}

    void memAlloc();
    void memcpyHtoD(const CUstream &stream);
    void memcpyDtoH(const CUstream &stream);
    void memFree();

private:
    int clockRate;
    long clock_count;
    long *h_a;
    CUdeviceptr d_a;
};

#endif

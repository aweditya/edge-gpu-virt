#ifndef _CLOCK_BLOCK_KERNEL_H
#define _CLOCK_BLOCK_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "clockBlock.h"
#include "Kernel.h"

class ClockBlockKernel : public Kernel
{
public:
    ClockBlockKernel(int clockRate, int **perSMThreads) : clockRate(clockRate),
                                                         perSMThreads(perSMThreads),
                                                         gridDimX(GRID_DIM_X),
                                                         gridDimY(GRID_DIM_Y),
                                                         gridDimZ(GRID_DIM_Z),
                                                         blockDimX(BLOCK_DIM_X),
                                                         blockDimY(BLOCK_DIM_Y),
                                                         blockDimZ(BLOCK_DIM_Z) {}
    ~ClockBlockKernel() {}

    void memAlloc();
    void memcpyHtoD(const CUstream &stream);
    void memcpyDtoH(const CUstream &stream);
    void memFree();

    void getKernelConfig(unsigned int &gridDimX, unsigned int &gridDimY, unsigned int &gridDimZ,
                         unsigned int &blockDimX, unsigned int &blockDimY, unsigned int &blockDimZ)
    {
        gridDimX = this->gridDimX;
        gridDimY = this->gridDimY;
        gridDimZ = this->gridDimZ;

        blockDimX = this->blockDimX;
        blockDimY = this->blockDimY;
        blockDimZ = this->blockDimZ;
    }

private:
    int clockRate;
    int gridDimX, gridDimY, gridDimZ;
    int blockDimX, blockDimY, blockDimZ;
    long clock_count;
    long *h_a;
    CUdeviceptr d_a;
    int **perSMThreads;
};

#endif

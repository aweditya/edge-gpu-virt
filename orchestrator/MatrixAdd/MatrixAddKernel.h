#ifndef _MATRIX_ADD_KERNEL_H
#define _MATRIX_ADD_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "matrixAdd.h"
#include "Kernel.h"

class MatrixAddKernel : public Kernel
{
public:
    MatrixAddKernel() {}
    ~MatrixAddKernel() {}

    void memAlloc();
    void memcpyHtoD(const CUstream &stream);
    void memcpyDtoH(const CUstream &stream);
    void memFree();

    void getKernelConfig(unsigned int &gridDimX, unsigned int &gridDimY, unsigned int &gridDimZ,
                         unsigned int &blockDimX, unsigned int &blockDimY, unsigned int &blockDimZ)
    {
        gridDimX = N;
        gridDimY = 1;
        gridDimZ = 1;

        blockDimX = 1;
        blockDimY = 1;
        blockDimZ = 1;
    }

private:
    double *h_a, *h_b, *h_c;
    CUdeviceptr d_a, d_b, d_c;
};

#endif

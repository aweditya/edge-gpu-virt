#ifndef _MATRIX_ADD_CALLBACK_H
#define _MATRIX_ADD_CALLBACK_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "matrixAdd.h"
#include "KernelCallback.h"

class MatrixAddCallback : public KernelCallback
{
public:
    MatrixAddCallback() {}
    ~MatrixAddCallback() {}

    void memAlloc();
    void memcpyHtoD(const cudaStream_t &stream);
    void memcpyDtoH(const cudaStream_t &stream);
    void memFree();

private:
    double *h_a, *h_b, *h_c;
    CUdeviceptr d_a, d_b, d_c;
};

#endif

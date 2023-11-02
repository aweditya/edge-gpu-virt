#ifndef _MATRIX_ADD_CALLBACK_H
#define _MATRIX_ADD_CALLBACK_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "matrixAdd.h"
#include "KernelCallback.h"

class MatrixAddCallback : public KernelCallback
{
public:
    MatrixAddCallback()
    {
        blockOffset = 0;
    }
    ~MatrixAddCallback() {}

    void memAlloc();
    void memcpyHtoD(const CUstream &stream);
    void memcpyDtoH(const CUstream &stream);
    void memFree();

private:
    double *h_a, *h_b, *h_c;
    CUdeviceptr d_a, d_b, d_c;
    int blockOffset;
};

#endif

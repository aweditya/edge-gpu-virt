#ifndef _KERNEL_CALLBACK_H
#define _KERNEL_CALLBACK_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "errchk.h"

class Kernel
{
public:
    Kernel() {}
    ~Kernel() {}

    virtual void memAlloc() = 0;
    virtual void memcpyHtoD(const CUstream &stream) = 0;
    virtual void memcpyDtoH(const CUstream &stream) = 0;
    virtual void memFree() = 0;
    virtual void getKernelConfig(unsigned int &gridDimX, unsigned int &gridDimY, unsigned int &gridDimZ,
                                 unsigned int &blockDimX, unsigned int &blockDimY, unsigned int &blockDimZ) = 0;

    void *args[16];
};

#endif

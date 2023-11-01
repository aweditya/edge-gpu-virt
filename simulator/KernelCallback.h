#ifndef _KERNEL_CALLBACK_H
#define _KERNEL_CALLBACK_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "errchk.h"

class KernelCallback
{
public:
    KernelCallback(CUstream &offeredStream)
    {
        stream = offeredStream;
    }
    ~KernelCallback();

    virtual void memAlloc() = 0;
    virtual void memcpyHtoD() = 0;
    virtual void memcpyDtoH() = 0;
    virtual void memFree() = 0;

protected:
    CUstream stream;
};

#endif 

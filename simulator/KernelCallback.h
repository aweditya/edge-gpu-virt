#ifndef _KERNEL_CALLBACK_H
#define _KERNEL_CALLBACK_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "errchk.h"

class KernelCallback
{
public:
    KernelCallback() {}
    ~KernelCallback() {}

    virtual void memAlloc() = 0;
    virtual void memcpyHtoD(const CUstream &stream) = 0;
    virtual void memcpyDtoH(const CUstream &stream) = 0;
    virtual void memFree() = 0;

    int getLauncherID() { return launcherID; }
    void setLauncherID(int id) { launcherID = id; }

    void *args[8];

protected:
    int launcherID;
};

#endif

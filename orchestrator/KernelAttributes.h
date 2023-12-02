#ifndef _KERNEL_ATTRIBUTES_H
#define _KERNEL_ATTRIBUTES_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "KernelControlBlock.h"

typedef struct kernel_attr
{
    unsigned int id;
    CUfunction function;

    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;

    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;

    unsigned int sGridDimX;
    unsigned int sGridDimY;
    unsigned int sGridDimZ;

    unsigned int blockOffsetX = 0;
    unsigned int blockOffsetY = 0;
    unsigned int blockOffsetZ = 0;

    unsigned int sharedMemBytes;
    CUstream stream;
    void **kernelParams;

    kernel_control_block_t kcb;

    unsigned int niceness = 0;
} kernel_attr_t;

struct compare_kernel_attr
{
    bool operator()(kernel_attr_t *attr_a, kernel_attr_t *attr_b) const
    {
        return attr_a->niceness > attr_b->niceness;
    }
};

#endif
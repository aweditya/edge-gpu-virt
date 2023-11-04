#ifndef _KERNEL_H
#define _KERNEL_H

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
} kernel_attr_t;

#endif
#ifndef _KERNEL_H
#define _KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

enum kstate
{
    INIT = 0,
    MEMCPYHTOD = 1,
    LAUNCH = 2,
    MEMCPYDTOH = 3
};

typedef struct kernel_control_block
{
    pthread_mutex_t kernel_lock;
    pthread_cond_t kernel_signal;
    kstate state;
    unsigned int slicesToLaunch;
    unsigned int totalSlices;
} kernel_control_block_t;

typedef struct kernel_attr
{
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
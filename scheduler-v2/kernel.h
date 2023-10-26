#pragma once

#include <cuda.h>

typedef struct
{
    dim3 blockConf;
    dim3 gridConf;
    const void *kernelFunction;
    void *arguments[16];
    size_t sharedMem;
    cudaStream_t *clientStream;
} kernel_args_t;

typedef struct
{
    kernel_args_t *kernel = nullptr;

    dim3 slicedGridConf;
    int totalSlices;
    int slicesToLaunch;
} kernel_metadata_t;

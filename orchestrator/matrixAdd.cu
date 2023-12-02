#include "matrixAdd.h"

extern "C" __global__ void matrixAdd(int blockOffsetX, int blockOffsetY, int blockOffsetZ, double *a, double *b, double *c)
{
    int tid = (blockIdx.x + blockOffsetX) * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}
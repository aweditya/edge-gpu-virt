#include "matrixAdd.h"

__global__ void matrxAdd(double *a, double *b, double *c, int blockOffset)
{
    int tid = (blockIdx.x + blockOffset) * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}
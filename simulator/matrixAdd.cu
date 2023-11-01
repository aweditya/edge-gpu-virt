#include "matrixAdd.h"

__global__ void matrxAdd(double *a, double *b, double *c, dim3 blockOffset)
{
    int tid = (blockIdx.x + blockOffset.x) * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}
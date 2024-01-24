#include "matrixAdd.h"
#include "SMIdentifier.h"

extern "C" __global__ void matrixAdd(int blockOffsetX, int blockOffsetY, int blockOffsetZ, 
                                     int blockDimX, int blockDimY, int blockDimZ,
                                     int *perSMThreads,
                                     double *a, double *b, double *c)
{
    int smID;
    if (threadIdx.x == 0 && threadIdx.y == 0 & threadIdx.z)
    {
        smID = get_smid();
        atomicAdd(&perSMThreads[smID], blockDimX * blockDimY * blockDimZ);
    }

    int tid = (blockIdx.x + blockOffsetX) * blockDim.x + threadIdx.x;
    if (tid < M)
    {
        c[tid] = a[tid] + b[tid];
    }

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z)
    {
        atomicSub(&perSMThreads[smID], blockDimX * blockDimY * blockDimZ);
    }
}

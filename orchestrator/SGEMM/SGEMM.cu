#include "SGEMM.h"
#include "SMIdentifier.h"

extern "C" __global__ void SGEMM(int blockOffsetX, int blockOffsetY, int blockOffsetZ,
                                 int blockDimX, int blockDimY, int blockDimZ,
                                 int *perSMThreads,
                                 const float *A, int lda,
                                 const float *B, int ldb,
                                 float *C, int ldc,
                                 int k, float alpha, float beta)
{
    int smID;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        smID = get_smid();
        atomicAdd(&perSMThreads[smID], blockDimX * blockDimY * blockDimZ);
    }

    // Partial results
    float c[TILE_N];
    for (int i = 0; i < TILE_N; i++)
        c[i] = 0.0f;
    int mid = threadIdx.y * blockDim.x + threadIdx.x; // flattened id
    int m = (blockOffsetX + blockIdx.x) * TILE_M + mid;
    int n = (blockOffsetY + blockIdx.y) * TILE_N + threadIdx.x;
    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];
    for (int i = 0; i < k; i += TILE_TB_HEIGHT)
    {
        float a;
        b_s[threadIdx.y][threadIdx.x] = B[n + (i + threadIdx.y) * ldb];
        __syncthreads();
        for (int j = 0; j < TILE_TB_HEIGHT; j++)
        {
            a = A[m + (i + j) * lda];
            for (int kk = 0; kk < TILE_N; kk++)
                c[kk] += a * b_s[j][kk];
        }
        __syncthreads();
    }
    int t = ldc * (blockOffsetY + blockIdx.y) * TILE_N + m;
    for (int i = 0; i < TILE_N; i++)
    {
        C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
    }

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        atomicSub(&perSMThreads[smID], blockDimX * blockDimY * blockDimZ);
    }
}

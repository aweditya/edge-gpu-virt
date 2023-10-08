/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * Kernel of dense matrix-matrix multiplication kernel.
 * The algorithm is based on CUDA sgemm code from Vasily Volkov
 * at UC Berkeley.
 */

#define CHECK_ERROR(errorMessage)                                               \
    {                                                                           \
        cudaError_t err = cudaGetLastError();                                   \
        if (cudaSuccess != err)                                                 \
        {                                                                       \
            fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",   \
                    errorMessage, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

// CML x RML = CML, baseline version, 510FLOP/s on Fermi
/* Pseudo code
for i < M ; i += 64   // thread block.x
 for j < N; j += 16   // thread block.y
  for tx = 0; tx < 16; tx++ // thread index x; tile of M loop
  for ty = 0; ty < 4 ; ty++ // thread index y; tile of M loop

  for m < 16; m += 1;
     c[m] = 0.0f

  for k < K; k += 4   // seq

   b[ty][tx] = B[k+ty][j+tx]

   for l < 4; l +=1   // seq
    for m < 16; m +=1 // seq
      c[m] += A[i+ty*16+tx][k+l]+b[l][m]

*/

// Parameters of tile sizes
#define TILE_N 16
#define TILE_TB_HEIGHT 8
#define TILE_M (TILE_N * TILE_TB_HEIGHT)

__global__ void mysgemmNT(const float *A, int lda, const float *B, int ldb, float *C, int ldc, int k, float alpha, float beta, dim3 blockOffset)
{
    // Partial results
    float c[TILE_N];
    for (int i = 0; i < TILE_N; i++)
        c[i] = 0.0f;
    int mid = threadIdx.y * blockDim.x + threadIdx.x; // flattened id
    int m = (blockOffset.x + blockIdx.x) * TILE_M + mid;
    int n = (blockOffset.y + blockIdx.y) * TILE_N + threadIdx.x;
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
    int t = ldc * (blockOffset.y + blockIdx.y) * TILE_N + m;
    for (int i = 0; i < TILE_N; i++)
    {
        C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
    }
}

void regtileSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, cudaStream_t *stream)
{
    if ((transa != 'N') && (transa != 'n'))
    {
        std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
        return;
    }

    if ((transb != 'T') && (transb != 't'))
    {
        std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
        return;
    }

    int m_slicer = 2, n_slicer = 6;
    // In this code we assume the matrix sizes are multiple of tile size
    if ((m % (TILE_M * m_slicer)) || (n % (TILE_N * n_slicer)))
    {
        std::cerr << "unsupported size of matrix. m should be multiple of " << TILE_M * m_slicer
                  << "; n should be multiple of " << TILE_N * n_slicer << std::endl;
    }

    dim3 gridConf(m / TILE_M, n / TILE_N);
    dim3 blockConf(TILE_N, TILE_TB_HEIGHT);
    dim3 sGridConf(m / (TILE_M * m_slicer), n / (TILE_N * n_slicer));

    printf("gridConf: (%d, %d)\n", gridConf.x, gridConf.y);
    printf("blockConf: (%d, %d)\n", blockConf.x, blockConf.y);
    printf("sGridConf: (%d, %d)\n", sGridConf.x, sGridConf.y);

    dim3 blockOffset(0, 0);
    while (blockOffset.x < m / TILE_M && blockOffset.y < n / TILE_N)
    {
        if (stream == nullptr)
            mysgemmNT<<<sGridConf, blockConf>>>(A, lda, B, ldb, C, ldc, k, alpha, beta, blockOffset);
        else
            mysgemmNT<<<sGridConf, blockConf, 0, *stream>>>(A, lda, B, ldb, C, ldc, k, alpha, beta, blockOffset);

        blockOffset.x += sGridConf.x;
        while (blockOffset.x >= gridConf.x)
        {
            blockOffset.x -= gridConf.x;
            blockOffset.y += sGridConf.y;
        }
    }

    CHECK_ERROR("mySgemm");
}

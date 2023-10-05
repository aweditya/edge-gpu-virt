#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <math.h>

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

__global__ void MatrixAdd(float *matA, float *matB, int width, dim3 blockOffset)
{
    int row = (blockIdx.x + blockOffset.x) * blockDim.x + threadIdx.x;
    int col = (blockIdx.y + blockOffset.y) * blockDim.y + threadIdx.y;

    matA[row * width + col] += matB[row * width + col];
}

float drand(float lo, float hi)
{
    return lo + (hi - lo) * rand() / RAND_MAX;
}

void *launch_kernel(void *thread_args)
{
    int width = 4;

    /* Grid dimension */
    dim3 gridConf(width, width);

    /* Block dimension */
    dim3 blockConf(width, width);

    int totalElements = gridConf.x * gridConf.y * blockConf.x * blockConf.y;
    printf("Matrices A and B of dimension (%d, %d) are being added\n", gridConf.x * blockConf.x, gridConf.y * blockConf.y);

    /* Allocate host memory for matA and matB */
    float *h_matA = (float *)malloc(totalElements * sizeof(float));
    float *h_matB = (float *)malloc(totalElements * sizeof(float));

    /* Initialize matA and matB randomly */
    for (int i = 0; i < totalElements; ++i)
    {
        h_matA[i] = drand(0.0, 1.0);
        h_matB[i] = drand(0.0, 1.0);
    }

    /* Allocate device memory for matA and matB */
    float *d_matA, *d_matB;

    if (!(cudaSuccess == cudaMalloc((void **)&d_matA, totalElements * sizeof(float)))) // allocating memory on GPU
    {
        printf("cuda malloc for d_matA failed!\n");
        CHECK_ERROR("cudaMalloc");
    }
    if (!(cudaSuccess == cudaMalloc((void **)&d_matB, totalElements * sizeof(float)))) // allocating memory on GPU
    {
        printf("cuda malloc for d_matB failed!\n");
        CHECK_ERROR("cudaMalloc");
    }

    /* copy data from host to device */
    if (!(cudaSuccess == cudaMemcpy(d_matA, h_matA, totalElements * sizeof(float), cudaMemcpyHostToDevice)))
    {
        printf("cudaMemcpyAsync for (d_matA, h_matA) failed!\n");
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpy(d_matB, h_matB, totalElements * sizeof(float), cudaMemcpyHostToDevice)))
    {
        printf("cudaMemcpyAsync for (d_matB, h_matB) failed!\n");
        CHECK_ERROR("cudaMemcpyAsync");
    }

    /* Sliced grid dimension: 8x1 */
    dim3 sGridConf(width / 4, width / 2);
    dim3 blockOffset(0, 0);
    while (blockOffset.x < gridConf.x && blockOffset.y < gridConf.y)
    {
        // printf("Calling slice with blockOffset (%d, %d)\n", blockOffset.x, blockOffset.y);
        MatrixAdd<<<sGridConf, blockConf>>>(d_matA, d_matB, width * width, blockOffset);
        blockOffset.x += sGridConf.x;
        while (blockOffset.x >= gridConf.x)
        {
            blockOffset.x -= gridConf.x;
            blockOffset.y += sGridConf.y;
        }
    }

    /* copy result from device to host */
    if (!(cudaSuccess == cudaMemcpy(h_matA, d_matA, totalElements * sizeof(float), cudaMemcpyDeviceToHost)))
    {
        printf("cudaMemcpy for (h_matA, d_matA) failed!\n");
        CHECK_ERROR("cudaMemcpy");
    }

    free(h_matA);
    free(h_matB);
    cudaFree(d_matA);
    cudaFree(d_matB);

    return NULL;
}

int main(int argc, char *argv[])
{
    srand(0);

    const int num_threads = 8;
    for (int i = 0; i < num_threads; ++i)
    {
        launch_kernel(NULL);
    }

    cudaDeviceReset();

    return 0;
}

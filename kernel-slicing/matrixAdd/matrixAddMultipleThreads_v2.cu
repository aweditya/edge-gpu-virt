#include <iostream>
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
    int width = 16;

    /* Grid dimension */
    dim3 gridConf(width, width);

    /* Block dimension */
    dim3 blockConf(width, width);

    /* Sliced grid dimension: 8x1 */
    dim3 sGridConf(width / 4, width / 2);

    int totalElements = gridConf.x * gridConf.y * blockConf.x * blockConf.y;
    int numKernels = (gridConf.x * gridConf.y) / (sGridConf.x * sGridConf.y);
    printf("Matrices A and B of dimension (%d, %d) are being added with %d kernels\n", gridConf.x * blockConf.x, gridConf.y * blockConf.y, numKernels);

    // Create a new stream for each kernel slice
    cudaStream_t *streams = (cudaStream_t *)malloc(numKernels * sizeof(cudaStream_t));

    for (int i = 0; i < numKernels; ++i)
    {
        if (!(cudaSuccess == cudaStreamCreate(&(streams[i]))))
        {
            CHECK_ERROR("cudaStreamCreate");
        }
    }

    /* Allocate host memory for matA and matB */
    float *h_matA, *h_matB;

    if (!(cudaSuccess == cudaMallocHost((void **)&h_matA, totalElements * sizeof(float)))) // allocating memory on CPU
    {
        CHECK_ERROR("cudaMallocHost");
    }

    if (!(cudaSuccess == cudaMallocHost((void **)&h_matB, totalElements * sizeof(float)))) // allocating memory on CPU
    {
        CHECK_ERROR("cudaMallocHost");
    }

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
        CHECK_ERROR("cudaMalloc");
    }
    if (!(cudaSuccess == cudaMalloc((void **)&d_matB, totalElements * sizeof(float)))) // allocating memory on GPU
    {
        CHECK_ERROR("cudaMalloc");
    }

    int currentKernel = 0; // Assigning each kernel slice to a unique stream
    int workPerKernel = totalElements / numKernels;
    dim3 blockOffset(0, 0);
    while (blockOffset.x < gridConf.x && blockOffset.y < gridConf.y)
    {
        // printf("Calling slice with blockOffset (%d, %d)\n", blockOffset.x, blockOffset.y);

        /* copy data from host to device */
        if (!(cudaSuccess == cudaMemcpyAsync(d_matA + currentKernel * workPerKernel, h_matA + currentKernel * workPerKernel, workPerKernel * sizeof(float), cudaMemcpyHostToDevice, streams[currentKernel])))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        if (!(cudaSuccess == cudaMemcpyAsync(d_matB + currentKernel * workPerKernel, h_matB + currentKernel * workPerKernel, workPerKernel * sizeof(float), cudaMemcpyHostToDevice, streams[currentKernel])))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        MatrixAdd<<<sGridConf, blockConf, 0, streams[currentKernel]>>>(d_matA, d_matB, width * width, blockOffset);
        blockOffset.x += sGridConf.x;
        while (blockOffset.x >= gridConf.x)
        {
            blockOffset.x -= gridConf.x;
            blockOffset.y += sGridConf.y;
        }

        /* copy result from device to host */
        if (!(cudaSuccess == cudaMemcpyAsync(h_matA + currentKernel * workPerKernel, d_matA + currentKernel * workPerKernel, workPerKernel * sizeof(float), cudaMemcpyDeviceToHost, streams[currentKernel])))
        {
            CHECK_ERROR("cudaMemcpy");
        }

        ++currentKernel;
    }

    for (int i = 0; i < numKernels; ++i)
    {
        if (!(cudaSuccess == cudaStreamDestroy(streams[i])))
        {
            CHECK_ERROR("cudaStreamDestroy");
        }
    }

    free(streams);
    cudaFreeHost(h_matA);
    cudaFreeHost(h_matB);
    cudaFree(d_matA);
    cudaFree(d_matB);

    return NULL;
}

int main(int argc, char *argv[])
{
    srand(0);
    float elapsed_time;

    const int num_threads = 8;
    pthread_t threads[num_threads];

    cudaEvent_t start_event, stop_event;
    if (!(cudaSuccess == cudaEventCreate(&start_event)))
    {
        CHECK_ERROR("cudaEventCreate");
    }

    if (!(cudaSuccess == cudaEventCreate(&stop_event)))
    {
        CHECK_ERROR("cudaEventCreate");
    }

    cudaEventRecord(start_event, 0);
    for (int i = 0; i < num_threads; ++i)
    {
        if (pthread_create(&threads[i], NULL, launch_kernel, NULL))
        {
            fprintf(stderr, "Error creating threadn");
            return 1;
        }
    }

    for (int i = 0; i < num_threads; ++i)
    {
        if (pthread_join(threads[i], NULL))
        {
            fprintf(stderr, "Error joining threadn");
            return 2;
        }
    }

    if (!(cudaSuccess == cudaEventRecord(stop_event, 0)))
    {
        CHECK_ERROR("cudaEventRecord");
    }

    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaDeviceReset();

    printf("Measured time for sample = %.3fms\n", elapsed_time);
    return 0;
}

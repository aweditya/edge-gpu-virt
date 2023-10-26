#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "kernel.h"
#define DEVICE_RESET cudaDeviceReset();

#define ITERS_PER_THREAD 8192

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
                file, line, static_cast<unsigned int>(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

typedef struct
{
    cudaStream_t stream;
    int offset;
    int slicer;
} targs_t;

int *d_a = 0;
int N = 4 * 1024;

__global__ void test_kernel(int *a, int offset, int N, dim3 blockOffset)
{
    int tid = threadIdx.x + blockDim.x * (blockIdx.x + blockOffset.x);
    if (tid < N)
    {
        for (int i = 0; i < ITERS_PER_THREAD; ++i)
        {
            a[tid] += offset;
        }
    }
}

void *thread_func(void *thread_args)
{
    struct timeval t0, t1, dt;

    targs_t *targs = (targs_t *)(thread_args);
    
    dim3 gridConf(4 * 1024);
    dim3 blockConf(1024);
    dim3 sGridConf;
    sGridConf.x = gridConf.x / targs->slicer;
    printf("(thread %ld) slices: %d\n", (long)pthread_self(), gridConf.x / sGridConf.x);
    printf("(thread %ld) griConf: %d\n", (long)pthread_self(), gridConf.x);
    printf("(thread %ld) blockConf: %d\n", (long)pthread_self(), blockConf.x);
    printf("(thread %ld) sGridConf: %d\n", (long)pthread_self(), sGridConf.x);

    dim3 blockOffset(0);

    gettimeofday(&t0, NULL);
    while (blockOffset.x < gridConf.x)
    {
        test_kernel<<<sGridConf, blockConf, 0, targs->stream>>>(d_a, targs->offset, N, blockOffset);
        blockOffset.x += sGridConf.x;
    }

    gettimeofday(&t1, NULL);
    timersub(&t1, &t0, &dt);
    printf("(thread %ld) thread_func took %ld.%06ld sec\n", (long)pthread_self(), dt.tv_sec, dt.tv_usec);

    return NULL;
}

int main()
{
    checkCudaErrors(cudaMalloc((void **)&d_a, N * sizeof(int)));
    checkCudaErrors(cudaMemset(d_a, 0, N * sizeof(int)));

    float elapsed = 0;

    targs_t targs[2];
    for (int i = 0; i < 2; ++i)
    {
        cudaStreamCreate(&(targs[i].stream));

        targs[i].offset = i + 1;
        targs[i].slicer = 4 * (i + 1);
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));

    thread_func(&(targs[0]));
    thread_func(&(targs[1]));
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));

    for (int i = 0; i < 2; ++i)
    {
        cudaStreamDestroy(targs[i].stream);
    }

    printf("measured time for sample = %.3fms\n", elapsed);

    int *a = (int *)malloc(N * sizeof(int));
    checkCudaErrors(cudaMemcpyAsync(a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost, 0));

    cudaFree(d_a);
    free(a);

    DEVICE_RESET

    return 0;
}

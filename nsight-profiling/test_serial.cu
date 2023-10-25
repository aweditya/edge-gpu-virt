#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "kernel.h"
#define DEVICE_RESET cudaDeviceReset();

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
    kernel_control_block_t kcb;

    cudaStream_t stream;
    int offset;
    int slicer;
} targs_t;

__global__ void test_kernel(int *a, int offset, int N, dim3 blockOffset)
{
    int tid = threadIdx.x + blockDim.x * (blockIdx.x + blockOffset.x);
    if (tid < N)
    {
        a[tid] += offset;
    }
}

void *thread_func(void *thread_args)
{
    struct timeval t0, t1, dt;

    targs_t *targs = (targs_t *)(thread_args);
    
    int N = 4 * 1024 * 1024;

    int *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&d_a, N * sizeof(int)));
    checkCudaErrors(cudaMemset(d_a, 0, N * sizeof(int)));

    dim3 gridConf(4 * 1024);
    dim3 blockConf(1024);
    dim3 sGridConf;
    sGridConf.x = gridConf.x / targs->slicer;
    targs->kcb.slices = targs->slicer;
    printf("(thread %ld) slices: %d\n", (long)pthread_self(), targs->kcb.slices);

    dim3 blockOffset(0);

    gettimeofday(&t0, NULL);
    while (blockOffset.x < gridConf.x)
    {
        pthread_mutex_lock(&(targs->kcb.kernel_lock));
        while (targs->kcb.state != RUNNING)
        {
            pthread_cond_wait(&(targs->kcb.kernel_signal), &(targs->kcb.kernel_lock));
        }
        test_kernel<<<sGridConf, blockConf, 0, targs->stream>>>(d_a, targs->offset, N, blockOffset);
        pthread_mutex_unlock(&(targs->kcb.kernel_lock));

        targs->kcb.slices--;
        targs->kcb.state = READY;
        blockOffset.x += sGridConf.x;
    }

    int *a = (int *)malloc(N * sizeof(int));
    checkCudaErrors(cudaMemcpyAsync(a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost, targs->stream));

    cudaFree(d_a);
    free(a);

    gettimeofday(&t1, NULL);
    timersub(&t1, &t0, &dt);
    printf("(thread %ld) thread_func took %ld.%06ld sec\n", (long)pthread_self(), dt.tv_sec, dt.tv_usec);

    return NULL;
}

int main()
{
    float elapsed = 0;

    pthread_t threads[2];
    targs_t targs[2];

    for (int i = 0; i < 2; ++i)
    {
        kernel_control_block_init(&(targs[i].kcb));

        cudaStreamCreate(&(targs[i].stream));
        targs[i].offset = i + 1;
        targs[i].slicer = 1;
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));

    pthread_create(&(threads[0]), NULL, thread_func, &(targs[0]));
    while (true)
    {
        pthread_mutex_lock(&(targs[0].kcb.kernel_lock));
        targs[0].kcb.state = RUNNING;
        pthread_cond_signal(&(targs[0].kcb.kernel_signal));
        pthread_mutex_unlock(&(targs[0].kcb.kernel_lock));

        if (targs[0].kcb.slices == 0)
            break;
    }
    pthread_join(threads[0], NULL);

    pthread_create(&(threads[1]), NULL, thread_func, &(targs[1]));
    while (true)
    {
        pthread_mutex_lock(&(targs[1].kcb.kernel_lock));
        targs[1].kcb.state = RUNNING;
        pthread_cond_signal(&(targs[1].kcb.kernel_signal));
        pthread_mutex_unlock(&(targs[1].kcb.kernel_lock));

        if (targs[1].kcb.slices == 0)
            break;
    }
    pthread_join(threads[1], NULL);


    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));

    for (int i = 0; i < 2; ++i)
    {
        kernel_control_block_destroy(&(targs[i].kcb));
        cudaStreamDestroy(targs[i].stream);
    }

    printf("measured time for sample = %.3fms\n", elapsed);
    DEVICE_RESET

    return 0;
}

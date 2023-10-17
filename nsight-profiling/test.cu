#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/time.h>

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
    cudaStream_t stream;
    int offset;
} targs_t;

__global__ void test_kernel(int *a, int offset, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N)
    {
        a[tid] += offset;
    }
}

void *thread_func(void *thread_args)
{
    struct timeval t0, t1, dt;
    gettimeofday(&t0, NULL);

    targs_t *targs = (targs_t *)(thread_args);
    
    int N = 4 * 1024 * 1024;

    int *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&d_a, N * sizeof(int)));
    checkCudaErrors(cudaMemset(d_a, 0, N * sizeof(int)));
    test_kernel<<<4 * 1024, 1024, 0, targs->stream>>>(d_a, targs->offset, N);

    int *a = (int *)malloc(N * sizeof(int));
    checkCudaErrors(cudaMemcpyAsync(a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost, targs->stream));

    cudaFree(d_a);
    free(a);

    gettimeofday(&t1, NULL);
    timersub(&t1, &t0, &dt);
    printf("thread_func (thread %ld) took %ld.%06ld sec\n", (long)pthread_self(), dt.tv_sec, dt.tv_usec);

    return NULL;
}

int main()
{
    float elapsed = 0;

    pthread_t threads[2];
    targs_t targs[2];

    for (int i = 0; i < 2; ++i)
    {
        cudaStreamCreate(&(targs[i].stream));
        targs[i].offset = i + 1;
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));
    for (int i = 0; i < 2; ++i)
    {
        if (pthread_create(&(threads[i]), NULL, thread_func, &(targs[i])))
        {
            fprintf(stderr, "Error creating threadn");
            return 1;
        }
    }

    for (int i = 0; i < 2; ++i)
    {
        if (pthread_join(threads[i], NULL))
        {
            fprintf(stderr, "Error joining threadn");
            return 2;
        }
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));

    for (int i = 0; i < 2; ++i)
    {
        cudaStreamDestroy(targs[i].stream);
    }

    printf("measured time for sample = %.3fms\n", elapsed);
    DEVICE_RESET

    return 0;
}

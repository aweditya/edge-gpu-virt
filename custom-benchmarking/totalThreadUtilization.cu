#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include "errchk.h"

#define NUM_KERNELS 2                                // Number of kernels
#define KERNEL_TIME 1000L                            // Time to run kernel for (in ms)
#define LOGGING_INTERVAL 10                          // Logging interval (in ms)
#define LOGGING_DURATION (NUM_KERNELS * KERNEL_TIME) // Logging duration (in ms)

typedef struct kernel_metadata
{
    int thread_args;
    unsigned int smID;
} kernel_metadata;

typedef struct kernel_thread_args
{
    kernel_metadata *metadata;
    long clock_count;
    cudaStream_t stream;
} kernel_thread_args;

// Get ID of SM on which kernel thread is running
__device__ unsigned int get_smid(void)
{
    unsigned int smID;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smID));
    return smID;
}

// This is a kernel that does no real work but runs at least for a specified number of clocks
__global__ void clockBlock(long clock_count, kernel_metadata *metadata)
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        atomicAdd(&metadata->thread_args, 1); // Increment when thread starts
        metadata->smID = get_smid();
    }

    unsigned int start_clock = (unsigned int)clock();

    long clock_offset = 0;
    while (clock_offset < clock_count)
    {
        unsigned int end_clock = (unsigned int)clock();

        // The code below should work like
        // this (thanks to modular arithmetics):
        //
        // clock_offset = (clock_t) (end_clock > start_clock ?
        //                           end_clock - start_clock :
        //                           end_clock + (0xffffffffu - start_clock));
        //
        // Indeed, let m = 2^32 then
        // end - start = end + m - start (mod m).

        clock_offset = (long)(end_clock - start_clock);
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        atomicSub(&metadata->thread_args, 1); // Decrement when thread starts
    }
}

// Periodically report number of threads for each kernel
__host__ void reportThreadsRunning(kernel_metadata *allMetadata)
{
    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        printf("(%u, %d)\t", allMetadata[i].smID, allMetadata[i].thread_args);
    }
    printf("\n");
}

void *kernel_threadfunction(void *args)
{
    kernel_thread_args *kernel_args = (kernel_thread_args *)args;
    clockBlock<<<32, 128, 0, kernel_args->stream>>>(kernel_args->clock_count, kernel_args->metadata);
    return NULL;
}

void *logger_threadfunction(void *args)
{
    kernel_metadata *allMetadata = (kernel_metadata *)args;
    const int iterations = LOGGING_DURATION / LOGGING_INTERVAL;
    for (int i = 0; i < iterations; ++i)
    {
        reportThreadsRunning(allMetadata);
        usleep(LOGGING_INTERVAL * 1000);
    }

    return NULL;
}

int main()
{
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    int clockRate = prop.clockRate;
    int clock_count = KERNEL_TIME * clockRate;

    // Allocate unified memory
    kernel_metadata *allMetadata = NULL;
    checkCudaErrors(cudaMallocManaged((void **)&allMetadata, sizeof(int) * NUM_KERNELS));

    kernel_thread_args args[NUM_KERNELS];
    pthread_t kernel_threads[NUM_KERNELS], logger_thread;

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        args[i].metadata = &allMetadata[i];
        args[i].clock_count = clock_count;
        checkCudaErrors(cudaStreamCreate(&args[i].stream));
    }

    /* Start logger thread */
    if (pthread_create(&logger_thread, NULL, logger_threadfunction, allMetadata) != 0)
    {
        perror("pthread_create failed");
    }

    /* Launch kernels */
    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        if (pthread_create(&kernel_threads[i], NULL, kernel_threadfunction, &args[i]) != 0)
        {
            perror("pthread_create failed");
        }
    }

    if (pthread_join(logger_thread, NULL) != 0)
    {
        perror("pthread_join failed");
    }

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        if (pthread_join(kernel_threads[i], NULL) != 0)
        {
            perror("pthread_join failed");
        }
    }

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        checkCudaErrors(cudaStreamDestroy(args[i].stream));
    }

    checkCudaErrors(cudaFree(allMetadata));
    checkCudaErrors(cudaDeviceReset());

    return 0;
}

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <parboil.h>
#include "sgemm_kernel.cu"
// #include "sgemm_kernel_sliced.cu"

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float> &);

extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

typedef struct thread_args
{
    cudaStream_t *stream;
    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT;
} thread_args_t;

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

void *launch_kernel(void *thread_args)
{
    float *dA, *dB, *dC;
    thread_args_t *args = (thread_args_t *)thread_args;

    // CUDA memory allocation
    std::vector<float> matC(args->matArow * args->matBcol);

    if (!(cudaSuccess == cudaMalloc((void **)&dA, args->A_sz)))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&dB, args->B_sz)))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&dC, args->C_sz)))
    {
        CHECK_ERROR("cudaMalloc");
    }

    // Copy A and B^T into device memory
    if (!(cudaSuccess == cudaMemcpyAsync(dA, &(args->matA.front()), args->A_sz, cudaMemcpyHostToDevice, *(args->stream))))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpyAsync(dB, &(args->matBT.front()), args->B_sz, cudaMemcpyHostToDevice, *(args->stream))))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    // Use standard sgemm interface
    regtileSgemm('N', 'T', args->matArow, args->matBcol, args->matAcol, 1.0f,
                 dA, args->matArow, dB, args->matBcol, 0.0f, dC, args->matArow, args->stream);

    if (!(cudaSuccess == cudaMemcpyAsync(&matC.front(), dC, args->C_sz, cudaMemcpyDeviceToHost, *(args->stream))))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return NULL;
}

int main(int argc, char *argv[])
{
    struct pb_Parameters *params;

    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT;

    /* Read command line. Expect 3 inputs: A, B and B^T
       in column-major layout*/
    params = pb_ReadParameters(&argc, argv);
    printf("%s %s %s\n", params->inpFiles[0], params->inpFiles[1], params->inpFiles[2]);
    if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] == NULL) || (params->inpFiles[2] == NULL) || (params->inpFiles[3] != NULL))
    {
        fprintf(stderr, "Expecting three input filenames\n");
        exit(-1);
    }

    /* Read in data */
    // load A
    readColMajorMatrixFile(params->inpFiles[0],
                           matArow, matAcol, matA);
    // copy A to device memory
    A_sz = matArow * matAcol * sizeof(float);

    // load B^T
    readColMajorMatrixFile(params->inpFiles[2],
                           matBcol, matBrow, matBT);

    B_sz = matBrow * matBcol * sizeof(float);

    // allocate space for C
    C_sz = matArow * matBcol * sizeof(float);

    float elapsed_time;

    const int num_threads = 4;
    pthread_t threads[num_threads];
    thread_args_t args[num_threads];

    cudaEvent_t start_event, stop_event;
    if (!(cudaSuccess == cudaEventCreate(&start_event)))
    {
        CHECK_ERROR("cudaEventCreate");
    }

    if (!(cudaSuccess == cudaEventCreate(&stop_event)))
    {
        CHECK_ERROR("cudaEventCreate");
    }

    for (int i = 0; i < num_threads; ++i)
    {
        if (!(cudaSuccess == cudaStreamCreate(args[i].stream)))
        {
            CHECK_ERROR("cudaStreamCreate");
        }
        args[i].A_sz = A_sz;
        args[i].B_sz = B_sz;
        args[i].C_sz = C_sz;
        args[i].matArow = matArow;
        args[i].matAcol = matAcol;
        args[i].matBrow = matBrow;
        args[i].matBcol = matBcol;
        args[i].matA = matA;
        args[i].matBT = matBT;
    }

    cudaEventRecord(start_event, 0);
    for (int i = 0; i < num_threads; ++i)
    {
        if (pthread_create(&threads[i], NULL, launch_kernel, &args[i]))
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

    for (int i = 0; i < num_threads; ++i)
        cudaStreamDestroy(*(args[i].stream));

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaDeviceReset();

    printf("Measured time for sample = %.3fms\n", elapsed_time);
    return 0;
}

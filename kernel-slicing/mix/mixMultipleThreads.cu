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
#include "sgemm_kernel_sliced.cu"
#include "computeQ_sliced.cu"
#include "file.h"

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float> &);

extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

typedef struct thread_args_sgemm
{
    cudaStream_t *stream;
    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT;
} thread_args_sgemm_t;

typedef struct thread_args_mriq
{
    cudaStream_t *stream;
    int numX, numK;      /* Number of X and K values */
    float *kx, *ky, *kz; /* K trajectory (3D vectors) */
    float *x, *y, *z;    /* X coordinates (3D vectors) */
    float *phiR, *phiI;  /* Phi values (complex) */
    float *phiMag;       /* Magnitude of Phi */
    float *Qr, *Qi;      /* Q signal (complex) */

} thread_args_mriq_t;

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

void *launch_kernel_sgemm(void *thread_args)
{
    float *dA, *dB, *dC;
    thread_args_sgemm_t *args = (thread_args_sgemm_t *)thread_args;

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

void *launch_kernel_mriq(void *thread_args)
{
    thread_args_mriq_t *args = (thread_args_mriq_t *)thread_args;

    /* Create CPU data structures */
    createDataStructsCPU(args->numK, args->numX, &(args->phiMag), &(args->Qr), &(args->Qi));

    /* GPU section 1 (precompute PhiMag) */
    {
        /* Mirror several data structures on the device */
        float *phiR_d, *phiI_d;
        float *phiMag_d;

        if (!(cudaSuccess == cudaMalloc((void **)&phiR_d, args->numK * sizeof(float))))
        {
            CHECK_ERROR("cudaMalloc");
        }

        if (!(cudaSuccess == cudaMalloc((void **)&phiI_d, args->numK * sizeof(float))))
        {
            CHECK_ERROR("cudaMalloc");
        }

        if (!(cudaSuccess == cudaMemcpyAsync(phiR_d, args->phiR, args->numK * sizeof(float), cudaMemcpyHostToDevice, *(args->stream))))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        if (!(cudaSuccess == cudaMemcpyAsync(phiI_d, args->phiI, args->numK * sizeof(float), cudaMemcpyHostToDevice, *(args->stream))))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        if (!(cudaSuccess == cudaMalloc((void **)&phiMag_d, args->numK * sizeof(float))))
        {
            CHECK_ERROR("cudaMalloc");
        }

        cudaDeviceSynchronize();

        computePhiMag_GPU(args->numK, phiR_d, phiI_d, phiMag_d, args->stream);

        cudaDeviceSynchronize();

        if (!(cudaSuccess == cudaMemcpyAsync(args->phiMag, phiMag_d, args->numK * sizeof(float), cudaMemcpyDeviceToHost, *(args->stream))))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        cudaFree(phiMag_d);
        cudaFree(phiR_d);
        cudaFree(phiI_d);
    }

    struct kValues *kVals;
    kVals = (struct kValues *)calloc(args->numK, sizeof(struct kValues));
    for (int k = 0; k < args->numK; k++)
    {
        kVals[k].Kx = args->kx[k];
        kVals[k].Ky = args->ky[k];
        kVals[k].Kz = args->kz[k];
        kVals[k].PhiMag = args->phiMag[k];
    }

    /* GPU section 2 */
    {
        float *x_d, *y_d, *z_d;
        float *Qr_d, *Qi_d;

        if (!(cudaSuccess == cudaMalloc((void **)&x_d, args->numX * sizeof(float))))
        {
            CHECK_ERROR("cudaMalloc");
        }

        if (!(cudaSuccess == cudaMemcpyAsync(x_d, args->x, args->numX * sizeof(float), cudaMemcpyHostToDevice, *(args->stream))))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        if (!(cudaSuccess == cudaMalloc((void **)&y_d, args->numX * sizeof(float))))
        {
            CHECK_ERROR("cudaMalloc");
        }

        if (!(cudaSuccess == cudaMemcpyAsync(y_d, args->y, args->numX * sizeof(float), cudaMemcpyHostToDevice, *(args->stream))))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        if (!(cudaSuccess == cudaMalloc((void **)&z_d, args->numX * sizeof(float))))
        {
            CHECK_ERROR("cudaMalloc");
        }

        if (!(cudaSuccess == cudaMemcpyAsync(z_d, args->z, args->numX * sizeof(float), cudaMemcpyHostToDevice, *(args->stream))))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        if (!(cudaSuccess == cudaMalloc((void **)&Qr_d, args->numX * sizeof(float))))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        if (!(cudaSuccess == cudaMemsetAsync((void *)Qr_d, 0, args->numX * sizeof(float), *(args->stream))))
        {
            CHECK_ERROR("cudaMemsetAsync");
        }

        if (!(cudaSuccess == cudaMalloc((void **)&Qi_d, args->numX * sizeof(float))))
        {
            CHECK_ERROR("cudaMalloc");
        }

        if (!(cudaSuccess == cudaMemsetAsync((void *)Qi_d, 0, args->numX * sizeof(float), *(args->stream))))
        {
            CHECK_ERROR("cudaMemsetAsync");
        }

        cudaDeviceSynchronize();

        computeQ_GPU(args->numK, args->numX, x_d, y_d, z_d, kVals, Qr_d, Qi_d, args->stream);

        cudaDeviceSynchronize();

        if (!(cudaSuccess == cudaMemcpyAsync(args->Qr, Qr_d, args->numX * sizeof(float), cudaMemcpyDeviceToHost, *(args->stream))))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        if (!(cudaSuccess == cudaMemcpyAsync(args->Qi, Qi_d, args->numX * sizeof(float), cudaMemcpyDeviceToHost, *(args->stream))))
        {
            CHECK_ERROR("cudaMemcpyAsync");
        }

        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(z_d);
        cudaFree(Qr_d);
        cudaFree(Qi_d);
    }

    free(kVals);
    return NULL;
}

int main(int argc, char *argv[])
{
    struct pb_Parameters *params;

    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT;

    int numX, numK;      /* Number of X and K values */
    int original_numK;   /* Number of K values in input file */
    float *kx, *ky, *kz; /* K trajectory (3D vectors) */
    float *x, *y, *z;    /* X coordinates (3D vectors) */
    float *phiR, *phiI;  /* Phi values (complex) */
    float *phiMag;       /* Magnitude of Phi */
    float *Qr, *Qi;      /* Q signal (complex) */

    /* Read command line. Expect 3 inputs: A, B and B^T
       in column-major layout*/
    params = pb_ReadParameters(&argc, argv);
    printf("%s %s %s\n", params->inpFiles[0], params->inpFiles[1], params->inpFiles[2]);
    if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] == NULL) || (params->inpFiles[2] == NULL) || (params->inpFiles[3] != NULL))
    {
        fprintf(stderr, "Expecting three input filenames\n");
        exit(-1);
    }

    inputData(params->inpFiles[0],
              &original_numK, &numX,
              &kx, &ky, &kz,
              &x, &y, &z,
              &phiR, &phiI);

    numK = original_numK;

    printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
           numX, original_numK, numK);

    /* Read in data */
    // load A
    readColMajorMatrixFile(params->inpFiles[1],
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
    thread_args_sgemm_t sgemm_args[num_threads / 2];
    thread_args_mriq_t mriq_args[num_threads / 2];

    cudaEvent_t start_event, stop_event;
    if (!(cudaSuccess == cudaEventCreate(&start_event)))
    {
        CHECK_ERROR("cudaEventCreate");
    }

    if (!(cudaSuccess == cudaEventCreate(&stop_event)))
    {
        CHECK_ERROR("cudaEventCreate");
    }

    for (int i = 0; i < num_threads / 2; ++i)
    {
        if (!(cudaSuccess == cudaStreamCreate(sgemm_args[i].stream)))
        {
            CHECK_ERROR("cudaStreamCreate");
        }
        sgemm_args[i].A_sz = A_sz;
        sgemm_args[i].B_sz = B_sz;
        sgemm_args[i].C_sz = C_sz;
        sgemm_args[i].matArow = matArow;
        sgemm_args[i].matAcol = matAcol;
        sgemm_args[i].matBrow = matBrow;
        sgemm_args[i].matBcol = matBcol;
        sgemm_args[i].matA = matA;
        sgemm_args[i].matBT = matBT;

        if (!(cudaSuccess == cudaStreamCreate(mriq_args[i].stream)))
        {
            CHECK_ERROR("cudaStreamCreate");
        }

        mriq_args[i].numX = numX;
        mriq_args[i].numK = numK;
        mriq_args[i].kx = kx;
        mriq_args[i].ky = ky;
        mriq_args[i].kz = kz;
        mriq_args[i].x = x;
        mriq_args[i].y = y;
        mriq_args[i].z = z;
        mriq_args[i].phiR = phiR;
        mriq_args[i].phiI = phiI;
        mriq_args[i].phiMag = phiMag;
        mriq_args[i].Qr = Qr;
        mriq_args[i].Qi = Qi;
    }

    cudaEventRecord(start_event, 0);
    for (int i = 0; i < num_threads / 2; ++i)
    {
        if (pthread_create(&threads[2 * i], NULL, launch_kernel_sgemm, &sgemm_args[i]))
        {
            fprintf(stderr, "Error creating threadn");
            return 1;
        }

        if (pthread_create(&threads[2 * i + 1], NULL, launch_kernel_mriq, &mriq_args[i]))
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

    for (int i = 0; i < num_threads / 2; ++i)
    {
        cudaStreamDestroy(*(sgemm_args[i].stream));
        cudaStreamDestroy(*(mriq_args[i].stream));
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaDeviceReset();

    printf("Measured time for sample = %.3fms\n", elapsed_time);

    // free(phiMag);
    // free(kx);
    // free(ky);
    // free(kz);
    // free(x);
    // free(y);
    // free(z);
    // free(phiR);
    // free(phiI);
    // free(Qr);
    // free(Qi);

    return 0;
}

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <parboil.h>
#include "sgemm_kernel_sliced.cu"
#include "computeQ_sliced.cu"
#include "file.h"
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

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float> &);

extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

void *launch_kernel_sgemm(void *thread_args)
{
    float *dA, *dB, *dC;
    sgemm_args_t *args = (sgemm_args_t *)thread_args;

    // CUDA memory allocation
    std::vector<float> matC(args->matArow * args->matBcol);

    checkCudaErrors(cudaMalloc((void **)&dA, args->A_sz));
    checkCudaErrors(cudaMalloc((void **)&dB, args->B_sz));
    checkCudaErrors(cudaMalloc((void **)&dC, args->C_sz));

    // Copy A and B^T into device memory
    checkCudaErrors(cudaMemcpyAsync(dA, &(args->matA.front()), args->A_sz, cudaMemcpyHostToDevice, args->stream));
    checkCudaErrors(cudaMemcpyAsync(dB, &(args->matBT.front()), args->B_sz, cudaMemcpyHostToDevice, args->stream));

    // Use standard sgemm interface
    regtileSgemm('N', 'T', args->matArow, args->matBcol, args->matAcol, 1.0f,
                 dA, args->matArow, dB, args->matBcol, 0.0f, dC, args->matArow, &(args->stream), args->kcb);

    checkCudaErrors(cudaMemcpyAsync(&matC.front(), dC, args->C_sz, cudaMemcpyDeviceToHost, args->stream));

    checkCudaErrors(cudaFree(dA));
    checkCudaErrors(cudaFree(dB));
    checkCudaErrors(cudaFree(dC));

    return NULL;
}

void *launch_kernel_mriq(void *thread_args)
{
    mriq_args_t *args = (mriq_args_t *)thread_args;

    /* Create CPU data structures */
    createDataStructsCPU(args->numK, args->numX, &(args->phiMag), &(args->Qr), &(args->Qi));

    /* GPU section 1 (precompute PhiMag) */
    {
        /* Mirror several data structures on the device */
        float *phiR_d, *phiI_d;
        float *phiMag_d;

        checkCudaErrors(cudaMalloc((void **)&phiR_d, args->numK * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&phiI_d, args->numK * sizeof(float)));

        checkCudaErrors(cudaMemcpyAsync(phiR_d, args->phiR, args->numK * sizeof(float), cudaMemcpyHostToDevice, args->stream));
        checkCudaErrors(cudaMemcpyAsync(phiI_d, args->phiI, args->numK * sizeof(float), cudaMemcpyHostToDevice, args->stream));

        checkCudaErrors(cudaMalloc((void **)&phiMag_d, args->numK * sizeof(float)));

        cudaDeviceSynchronize();

        computePhiMag_GPU(args->numK, phiR_d, phiI_d, phiMag_d, &(args->stream), args->kcb);

        cudaDeviceSynchronize();

        checkCudaErrors(cudaMemcpyAsync(args->phiMag, phiMag_d, args->numK * sizeof(float), cudaMemcpyDeviceToHost, args->stream));

        checkCudaErrors(cudaFree(phiMag_d));
        checkCudaErrors(cudaFree(phiR_d));
        checkCudaErrors(cudaFree(phiI_d));
    }

    // struct kValues *kVals;
    // kVals = (struct kValues *)calloc(args->numK, sizeof(struct kValues));
    // for (int k = 0; k < args->numK; k++)
    // {
    //     kVals[k].Kx = args->kx[k];
    //     kVals[k].Ky = args->ky[k];
    //     kVals[k].Kz = args->kz[k];
    //     kVals[k].PhiMag = args->phiMag[k];
    // }

    // /* GPU section 2 */
    // {
    //     float *x_d, *y_d, *z_d;
    //     float *Qr_d, *Qi_d;

    //     checkCudaErrors(cudaMalloc((void **)&x_d, args->numX * sizeof(float)));
    //     checkCudaErrors(cudaMemcpyAsync(x_d, args->x, args->numX * sizeof(float), cudaMemcpyHostToDevice, args->stream));

    //     checkCudaErrors(cudaMalloc((void **)&y_d, args->numX * sizeof(float)));
    //     checkCudaErrors(cudaMemcpyAsync(y_d, args->y, args->numX * sizeof(float), cudaMemcpyHostToDevice, args->stream));

    //     checkCudaErrors(cudaMalloc((void **)&z_d, args->numX * sizeof(float)));
    //     checkCudaErrors(cudaMemcpyAsync(z_d, args->z, args->numX * sizeof(float), cudaMemcpyHostToDevice, args->stream));

    //     checkCudaErrors(cudaMalloc((void **)&Qr_d, args->numX * sizeof(float)));
    //     checkCudaErrors(cudaMemsetAsync((void *)Qr_d, 0, args->numX * sizeof(float), args->stream));

    //     checkCudaErrors(cudaMalloc((void **)&Qi_d, args->numX * sizeof(float)));
    //     checkCudaErrors(cudaMemsetAsync((void *)Qi_d, 0, args->numX * sizeof(float), args->stream));

    //     cudaDeviceSynchronize();

    //     computeQ_GPU(args->numK, args->numX, x_d, y_d, z_d, kVals, Qr_d, Qi_d, &args->stream, args->kcb);

    //     cudaDeviceSynchronize();

    //     checkCudaErrors(cudaMemcpyAsync(args->Qr, Qr_d, args->numX * sizeof(float), cudaMemcpyDeviceToHost, args->stream));
    //     checkCudaErrors(cudaMemcpyAsync(args->Qi, Qi_d, args->numX * sizeof(float), cudaMemcpyDeviceToHost, args->stream));

    //     checkCudaErrors(cudaFree(x_d));
    //     checkCudaErrors(cudaFree(y_d));
    //     checkCudaErrors(cudaFree(z_d));
    //     checkCudaErrors(cudaFree(Qr_d));
    //     checkCudaErrors(cudaFree(Qi_d));
    // }

    // free(kVals);
    return NULL;
}

int main(int argc, char **argv)
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

    const int num_threads = 2;
    pthread_t threads[num_threads];
    sgemm_args_t sgemm_args[num_threads / 2];
    mriq_args_t mriq_args[num_threads / 2];

    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    for (int i = 0; i < num_threads / 2; ++i)
    {
        kernel_control_block_init(sgemm_args[i].kcb);
        checkCudaErrors(cudaStreamCreate(&(sgemm_args[i].stream)));
        sgemm_args[i].A_sz = A_sz;
        sgemm_args[i].B_sz = B_sz;
        sgemm_args[i].C_sz = C_sz;
        sgemm_args[i].matArow = matArow;
        sgemm_args[i].matAcol = matAcol;
        sgemm_args[i].matBrow = matBrow;
        sgemm_args[i].matBcol = matBcol;
        sgemm_args[i].matA = matA;
        sgemm_args[i].matBT = matBT;

        kernel_control_block_init(mriq_args[i].kcb);
        checkCudaErrors(cudaStreamCreate(&(mriq_args[i].stream)));
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

    int launch = 1;
    while (launch)
    {
        printf("Launch (%d)\n", launch++);

        if (launch % 3 == 0)
        {
            printf("Let sgemm run\n");
            pthread_mutex_lock(&(sgemm_args[0].kcb->kernel_lock));
            printf("Got the sgeem lock\n");
            sgemm_args[0].kcb->state = RUNNING;
            pthread_mutex_unlock(&(sgemm_args[0].kcb->kernel_lock));
            printf("Released the sgeem lock\n");
            pthread_cond_signal(&(sgemm_args[0].kcb->kernel_signal));
            printf("sgemm signalled\n");
        }
        else
        {
            printf("Let mriq run\n");
            pthread_mutex_lock(&(mriq_args[0].kcb->kernel_lock));
            printf("Got the mriq lock\n");
            mriq_args[0].kcb->state = RUNNING;
            pthread_mutex_unlock(&(mriq_args[0].kcb->kernel_lock));
            printf("Released the mriq lock\n");
            pthread_cond_signal(&(mriq_args[0].kcb->kernel_signal));
            printf("mriq signalled\n");
        }

        if (sgemm_args[0].kcb->slices == 0 && mriq_args[0].kcb->slices == 0)
            launch = 0;
    }

    for (int i = 0; i < num_threads; ++i)
    {
        if (pthread_join(threads[i], NULL))
        {
            fprintf(stderr, "Error joining threadn");
            return 2;
        }
    }

    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

    for (int i = 0; i < num_threads / 2; ++i)
    {
        kernel_control_block_destroy(sgemm_args[i].kcb);
        kernel_control_block_destroy(mriq_args[i].kcb);

        cudaStreamDestroy(sgemm_args[i].stream);
        cudaStreamDestroy(mriq_args[i].stream);
    }

    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));

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
    DEVICE_RESET

    return 0;
}
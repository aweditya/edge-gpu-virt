/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * C code for creating the Q data structure for fast convolution-based
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <malloc.h>
#include <string.h>
#include <vector>
#include <parboil.h>
#include <iostream>
#include "file.h"
// #include "computeQ.cu"
// #include "sgemm_kernel.cu"
#include "computeQ_sliced.cu"
#include "sgemm_kernel_sliced.cu"

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float> &);

extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

static void
setupMemoryGPU(int num, int size, float *&dev_ptr, float *&host_ptr)
{
    cudaMalloc((void **)&dev_ptr, num * size);
    CUDA_ERRCK;
    cudaMemcpy(dev_ptr, host_ptr, num * size, cudaMemcpyHostToDevice);
    CUDA_ERRCK;
}

static void
cleanupMemoryGPU(int num, int size, float *&dev_ptr, float *host_ptr)
{
    cudaMemcpy(host_ptr, dev_ptr, num * size, cudaMemcpyDeviceToHost);
    CUDA_ERRCK;
    cudaFree(dev_ptr);
    CUDA_ERRCK;
}

int main(int argc, char *argv[])
{
    int numX, numK;      /* Number of X and K values */
    int original_numK;   /* Number of K values in input file */
    float *kx, *ky, *kz; /* K trajectory (3D vectors) */
    float *x, *y, *z;    /* X coordinates (3D vectors) */
    float *phiR, *phiI;  /* Phi values (complex) */
    float *phiMag;       /* Magnitude of Phi */
    float *Qr, *Qi;      /* Q signal (complex) */

    float *dA, *dB, *dC;
    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT;

    struct kValues *kVals;

    struct pb_Parameters *params;
    struct pb_TimerSet timers;

    pb_InitializeTimerSet(&timers);

    /* Read command line */
    params = pb_ReadParameters(&argc, argv);

    /* Read in data */
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    inputData(params->inpFiles[0],
              &original_numK, &numX,
              &kx, &ky, &kz,
              &x, &y, &z,
              &phiR, &phiI);

    // load A
    readColMajorMatrixFile(params->inpFiles[1],
                           matArow, matAcol, matA);
    // copy A to device memory
    A_sz = matArow * matAcol * sizeof(float);

    // load B^T
    readColMajorMatrixFile(params->inpFiles[2],
                           matBcol, matBrow, matBT);

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    B_sz = matBrow * matBcol * sizeof(float);

    // allocate space for C
    C_sz = matArow * matBcol * sizeof(float);

    // CUDA memory allocation
    std::vector<float> matC(matArow * matBcol);
    cudaMalloc((void **)&dA, A_sz);
    cudaMalloc((void **)&dB, B_sz);
    cudaMalloc((void **)&dC, C_sz);

    // Copy A and B^T into device memory
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    cudaMemcpy(dA, &matA.front(), A_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &matBT.front(), B_sz, cudaMemcpyHostToDevice);

    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

    // Use standard sgemm interface
    for (int i = 0; i < 1; ++i)
    {
        printf("Iteration %d\n", i);
        regtileSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f,
                     dA, matArow, dB, matBcol, 0.0f, dC, matArow);

        pb_SwitchToTimer(&timers, pb_TimerID_NONE);

        double GPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_KERNEL]));
        std::cout << "GFLOPs = " << 2. * matArow * matBcol * matAcol / GPUtime / 1e9 << std::endl;
    }

    /* Reduce the number of k-space samples if a number is given
     * on the command line */
    numK = original_numK;

    printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
           numX, original_numK, numK);

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    /* Create CPU data structures */
    createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

    /* GPU section 1 (precompute PhiMag) */
    {
        /* Mirror several data structures on the device */
        float *phiR_d, *phiI_d;
        float *phiMag_d;

        pb_SwitchToTimer(&timers, pb_TimerID_COPY);
        setupMemoryGPU(numK, sizeof(float), phiR_d, phiR);
        setupMemoryGPU(numK, sizeof(float), phiI_d, phiI);
        cudaMalloc((void **)&phiMag_d, numK * sizeof(float));
        CUDA_ERRCK;

        cudaThreadSynchronize();
        pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

        computePhiMag_GPU(numK, phiR_d, phiI_d, phiMag_d);

        cudaThreadSynchronize();
        pb_SwitchToTimer(&timers, pb_TimerID_COPY);

        cleanupMemoryGPU(numK, sizeof(float), phiMag_d, phiMag);
        cudaFree(phiR_d);
        cudaFree(phiI_d);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    kVals = (struct kValues *)calloc(numK, sizeof(struct kValues));
    for (int k = 0; k < numK; k++)
    {
        kVals[k].Kx = kx[k];
        kVals[k].Ky = ky[k];
        kVals[k].Kz = kz[k];
        kVals[k].PhiMag = phiMag[k];
    }

    free(phiMag);

    /* GPU section 2 */
    {
        float *x_d, *y_d, *z_d;
        float *Qr_d, *Qi_d;

        pb_SwitchToTimer(&timers, pb_TimerID_COPY);

        setupMemoryGPU(numX, sizeof(float), x_d, x);
        setupMemoryGPU(numX, sizeof(float), y_d, y);
        setupMemoryGPU(numX, sizeof(float), z_d, z);
        cudaMalloc((void **)&Qr_d, numX * sizeof(float));
        CUDA_ERRCK;
        cudaMemset((void *)Qr_d, 0, numX * sizeof(float));
        cudaMalloc((void **)&Qi_d, numX * sizeof(float));
        CUDA_ERRCK;
        cudaMemset((void *)Qi_d, 0, numX * sizeof(float));

        cudaThreadSynchronize();
        pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

        computeQ_GPU(numK, numX, x_d, y_d, z_d, kVals, Qr_d, Qi_d);

        cudaThreadSynchronize();
        pb_SwitchToTimer(&timers, pb_TimerID_COPY);

        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(z_d);
        cleanupMemoryGPU(numX, sizeof(float), Qr_d, Qr);
        cleanupMemoryGPU(numX, sizeof(float), Qi_d, Qi);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    if (params->outFile)
    {
        /* Write Q to file */
        pb_SwitchToTimer(&timers, pb_TimerID_IO);
        outputData(params->outFile, Qr, Qi, numX);
        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

    free(kx);
    free(ky);
    free(kz);
    free(x);
    free(y);
    free(z);
    free(phiR);
    free(phiI);
    free(kVals);
    free(Qr);
    free(Qi);

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);

    pb_FreeParameters(params);

    return 0;
}

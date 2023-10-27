#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
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
    cudaStream_t stream;
    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT;
} thread_args_sgemm_t;

typedef struct thread_args_mriq
{
    cudaStream_t stream;
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

struct timeval t0, t1, dt;

float *dA, *dB, *dC;

float *phiR_d, *phiI_d;
float *phiMag_d;
float *x_d, *y_d, *z_d;
float *Qr_d, *Qi_d;

int main(int argc, char *argv[])
{
    float elapsed_time;

    cudaEvent_t start_event, stop_event;
    if (!(cudaSuccess == cudaEventCreate(&start_event)))
    {
        CHECK_ERROR("cudaEventCreate");
    }

    if (!(cudaSuccess == cudaEventCreate(&stop_event)))
    {
        CHECK_ERROR("cudaEventCreate");
    }

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

    /* Memory allocation for SGEMM */
    if (!(cudaSuccess == cudaMalloc((void **)&dA, A_sz)))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&dB, B_sz)))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&dC, C_sz)))
    {
        CHECK_ERROR("cudaMalloc");
    }
    /*******************************/

    /* Memory allocation for MRI-Q */

    /* Create CPU data structures */
    createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

    if (!(cudaSuccess == cudaMalloc((void **)&phiR_d, numK * sizeof(float))))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&phiI_d, numK * sizeof(float))))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&phiMag_d, numK * sizeof(float))))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&x_d, numX * sizeof(float))))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&y_d, numX * sizeof(float))))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&z_d, numX * sizeof(float))))
    {
        CHECK_ERROR("cudaMalloc");
    }

    if (!(cudaSuccess == cudaMalloc((void **)&Qr_d, numX * sizeof(float))))
    {
        CHECK_ERROR("cudaMalloc");
    }
    if (!(cudaSuccess == cudaMalloc((void **)&Qi_d, numX * sizeof(float))))
    {
        CHECK_ERROR("cudaMalloc");
    }
    /******************************/

    thread_args_sgemm_t sgemm_args;
    thread_args_mriq_t mriq_args;
    if (!(cudaSuccess == cudaStreamCreate(&sgemm_args.stream)))
    {
        CHECK_ERROR("cudaStreamCreate");
    }
    sgemm_args.A_sz = A_sz;
    sgemm_args.B_sz = B_sz;
    sgemm_args.C_sz = C_sz;
    sgemm_args.matArow = matArow;
    sgemm_args.matAcol = matAcol;
    sgemm_args.matBrow = matBrow;
    sgemm_args.matBcol = matBcol;
    sgemm_args.matA = matA;
    sgemm_args.matBT = matBT;

    if (!(cudaSuccess == cudaStreamCreate(&mriq_args.stream)))
    {
        CHECK_ERROR("cudaStreamCreate");
    }

    mriq_args.numX = numX;
    mriq_args.numK = numK;
    mriq_args.kx = kx;
    mriq_args.ky = ky;
    mriq_args.kz = kz;
    mriq_args.x = x;
    mriq_args.y = y;
    mriq_args.z = z;
    mriq_args.phiR = phiR;
    mriq_args.phiI = phiI;
    mriq_args.phiMag = phiMag;
    mriq_args.Qr = Qr;
    mriq_args.Qi = Qi;

    // CUDA memory allocation
    std::vector<float> matC(matArow * matBcol);

    cudaEventRecord(start_event, 0);
    /* SGEMM host to device memory copy */
    // Copy A and B^T into device memory
    if (!(cudaSuccess == cudaMemcpyAsync(dA, &(sgemm_args.matA.front()), sgemm_args.A_sz, cudaMemcpyHostToDevice, sgemm_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpyAsync(dB, &(sgemm_args.matBT.front()), sgemm_args.B_sz, cudaMemcpyHostToDevice, sgemm_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    /* MRI-Q host to device memory copy */
    /* Mirror several data structures on the device */
    if (!(cudaSuccess == cudaMemcpyAsync(phiR_d, mriq_args.phiR, mriq_args.numK * sizeof(float), cudaMemcpyHostToDevice, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpyAsync(phiI_d, mriq_args.phiI, mriq_args.numK * sizeof(float), cudaMemcpyHostToDevice, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpyAsync(x_d, mriq_args.x, mriq_args.numX * sizeof(float), cudaMemcpyHostToDevice, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpyAsync(y_d, mriq_args.y, mriq_args.numX * sizeof(float), cudaMemcpyHostToDevice, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpyAsync(z_d, mriq_args.z, mriq_args.numX * sizeof(float), cudaMemcpyHostToDevice, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemsetAsync((void *)Qr_d, 0, mriq_args.numX * sizeof(float), mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemsetAsync");
    }

    if (!(cudaSuccess == cudaMemsetAsync((void *)Qi_d, 0, mriq_args.numX * sizeof(float), mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemsetAsync");
    }

    /* MRI-Q PhiMag computation */
    int phiMagBlocks = mriq_args.numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
    if (mriq_args.numK % KERNEL_PHI_MAG_THREADS_PER_BLOCK)
        phiMagBlocks++;

    dim3 gridConf(phiMagBlocks, 1);
    dim3 blockConf(KERNEL_PHI_MAG_THREADS_PER_BLOCK, 1);
    dim3 blockOffset(0);
    ComputePhiMag_GPU<<<gridConf, blockConf, 0, mriq_args.stream>>>(phiR_d, phiI_d, phiMag_d, mriq_args.numK, blockOffset);
    /* MRI-Q PhiMag computation */

    /* Some additional setup for MRI-Q */
    struct kValues *kVals;
    kVals = (struct kValues *)calloc(mriq_args.numK, sizeof(struct kValues));
    for (int k = 0; k < mriq_args.numK; k++)
    {
        kVals[k].Kx = mriq_args.kx[k];
        kVals[k].Ky = mriq_args.ky[k];
        kVals[k].Kz = mriq_args.kz[k];
        kVals[k].PhiMag = mriq_args.phiMag[k];
    }

    int QGridBase1 = 0, QGridBase2 = KERNEL_Q_K_ELEMS_PER_GRID;
    kValues *kValsTile1 = kVals + QGridBase1, *kValsTile2 = kVals + QGridBase2;
    int numElems1 = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase1), numElems2 = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase2);

    if (!(cudaSuccess == cudaMemcpyToSymbolAsync(ck, kValsTile1, numElems1 * sizeof(kValues), 0, cudaMemcpyHostToDevice, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyToSymbolAsync");
    }

    if (!(cudaSuccess == cudaMemcpyToSymbolAsync(ck, kValsTile2, numElems2 * sizeof(kValues), 0, cudaMemcpyHostToDevice, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyToSymbolAsync");
    }
    /***********************************/

    // Use standard sgemm interface
    int m = matArow;
    int n = matBcol;
    int k = matAcol;
    float alpha = 1.0f, beta = 0.0f;
    int lda = matArow;
    int ldb = matBcol;
    int ldc = matArow;

    int m_slicer = 1, n_slicer = 1;
    dim3 sgemmGridConf(m / TILE_M, n / TILE_N);
    dim3 sgemmBlockConf(TILE_N, TILE_TB_HEIGHT);
    dim3 sgemmSGridConf(m / (TILE_M * m_slicer), n / (TILE_N * n_slicer));
    int sgemmTotalSlices = m_slicer * n_slicer;

    int QGrids = mriq_args.numK / KERNEL_Q_K_ELEMS_PER_GRID;
    if (mriq_args.numK % KERNEL_Q_K_ELEMS_PER_GRID)
        QGrids++;
    int QBlocks = mriq_args.numX / KERNEL_Q_THREADS_PER_BLOCK;
    if (mriq_args.numX % KERNEL_Q_THREADS_PER_BLOCK)
        QBlocks++;

    int slicer = 1;
    dim3 mriqGridConf(QBlocks, 1);
    dim3 mriqBlockConf(KERNEL_Q_THREADS_PER_BLOCK, 1);
    dim3 mriqSGridConf(QBlocks / slicer, 1);
    int mriq1TotalSlices = slicer, mriq2TotalSlices = slicer;

    /* Printing kernel information */
    printf("(SGEMM) gridConf: (%d, %d)\n", sgemmGridConf.x, sgemmGridConf.y);
    printf("(SGEMM) blockConf: (%d, %d)\n", sgemmBlockConf.x, sgemmBlockConf.y);
    printf("(SGEMM) sGridConf: (%d, %d)\n", sgemmSGridConf.x, sgemmSGridConf.y);
    printf("(SGEMM) number of slices: %d\n", m_slicer * n_slicer);

    printf("(MRI-Q x2) gridConf: (%d, %d)\n", mriqGridConf.x, mriqGridConf.y);
    printf("(MRI-Q x2) blockConf: (%d, %d)\n", mriqBlockConf.x, mriqBlockConf.y);
    printf("(MRI-Q x2) sGridConf: (%d, %d)\n", mriqSGridConf.x, mriqSGridConf.y);
    printf("(MRI-Q x2) number of slices: %d\n", slicer);
    /*******************************/

    dim3 sgemmBlockOffset(0, 0);
    dim3 mriq1BlockOffset(0), mriq2BlockOffset(0);

    int launch = 1;
    while (launch)
    {
        if (launch % 3 == 1)
        {
            if (sgemmTotalSlices)
            {
                for (int i = 0; i < 1; ++i)
                {
                    mysgemmNT<<<sgemmSGridConf, sgemmBlockConf, 0, sgemm_args.stream>>>(dA, lda, dB, ldb, dC, ldc, k, alpha, beta, sgemmBlockOffset);

                    sgemmBlockOffset.x += sgemmSGridConf.x;
                    while (sgemmBlockOffset.x >= sgemmGridConf.x)
                    {
                        sgemmBlockOffset.x -= sgemmGridConf.x;
                        sgemmBlockOffset.y += sgemmSGridConf.y;
                    }

                    sgemmTotalSlices--;
                }
            }
        }
        else if (launch % 3 == 2)
        {
            if (mriq1TotalSlices)
            {
                ComputeQ_GPU<<<mriqSGridConf, mriqBlockConf, 0, mriq_args.stream>>>(mriq_args.numK, QGridBase1, x_d, y_d, z_d, Qr_d, Qi_d, mriq1BlockOffset);
                mriq1BlockOffset.x += mriqSGridConf.x;

                mriq1TotalSlices--;
            }
        }
        else
        {
            if (mriq2TotalSlices)
            {
                ComputeQ_GPU<<<mriqSGridConf, mriqBlockConf, 0, mriq_args.stream>>>(mriq_args.numK, QGridBase2, x_d, y_d, z_d, Qr_d, Qi_d, mriq2BlockOffset);
                mriq2BlockOffset.x += mriqSGridConf.x;

                mriq2TotalSlices--;
            }
        }

        launch++;
        if (sgemmTotalSlices == 0 && mriq1TotalSlices == 0 && mriq2TotalSlices == 0)
        {
            launch = 0;
        }
    }

    if (!(cudaSuccess == cudaMemcpyAsync(&matC.front(), dC, sgemm_args.C_sz, cudaMemcpyDeviceToHost, sgemm_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpyAsync(mriq_args.phiMag, phiMag_d, mriq_args.numK * sizeof(float), cudaMemcpyDeviceToHost, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpyAsync(mriq_args.Qr, Qr_d, mriq_args.numX * sizeof(float), cudaMemcpyDeviceToHost, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaMemcpyAsync(mriq_args.Qi, Qi_d, mriq_args.numX * sizeof(float), cudaMemcpyDeviceToHost, mriq_args.stream)))
    {
        CHECK_ERROR("cudaMemcpyAsync");
    }

    if (!(cudaSuccess == cudaEventRecord(stop_event, 0)))
    {
        CHECK_ERROR("cudaEventRecord");
    }

    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

    cudaStreamDestroy(sgemm_args.stream);
    cudaStreamDestroy(mriq_args.stream);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    /* Free SGEMM memory */
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    /*********************/

    /* Free MRI-Q memory */
    cudaFree(phiMag_d);
    cudaFree(phiR_d);
    cudaFree(phiI_d);

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaFree(Qr_d);
    cudaFree(Qi_d);
    /*********************/

    cudaDeviceReset();

    printf("Measured time for sample = %.3fms\n", elapsed_time);

    free(kVals);
    free(phiMag);
    free(kx);
    free(ky);
    free(kz);
    free(x);
    free(y);
    free(z);
    free(phiR);
    free(phiI);
    free(Qr);
    free(Qi);

    return 0;
}

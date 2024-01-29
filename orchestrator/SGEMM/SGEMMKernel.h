#ifndef _SGEMM_KERNEL_H
#define _SGEMM_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <parboil.h>
#include <vector>

#include "SGEMM.h"
#include "Kernel.h"

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v);

class SGEMMKernel : public Kernel
{
public:
    SGEMMKernel(struct pb_Parameters *params, int **perSMThreads) : params(params),
                                                                    perSMThreads(perSMThreads),
                                                                    blockDimX(TILE_N),
                                                                    blockDimY(TILE_TB_HEIGHT),
                                                                    blockDimZ(1)
    {
        alpha = 1.0f;
        beta = 1.0f;

        readColMajorMatrixFile(params->inpFiles[0], matArow, matAcol, matA);
        readColMajorMatrixFile(params->inpFiles[1], matBcol, matBrow, matBT);

        A_sz = matArow * matAcol * sizeof(float);
        B_sz = matBrow * matBcol * sizeof(float);
        C_sz = matArow * matBcol * sizeof(float);

        gridDimX = matArow / TILE_M;
        gridDimY = matBcol / TILE_N;
        gridDimZ = 1;
    }
    ~SGEMMKernel() {}

    void memAlloc();
    void memcpyHtoD(const CUstream &stream);
    void memcpyDtoH(const CUstream &stream);
    void memFree();

    void getKernelConfig(unsigned int &gridDimX, unsigned int &gridDimY, unsigned int &gridDimZ,
                         unsigned int &blockDimX, unsigned int &blockDimY, unsigned int &blockDimZ)
    {
        gridDimX = this->gridDimX;
        gridDimY = this->gridDimY;
        gridDimZ = this->gridDimZ;

        blockDimX = this->blockDimX;
        blockDimY = this->blockDimY;
        blockDimZ = this->blockDimZ;
    }

private:
    struct pb_Parameters *params;
    int gridDimX, gridDimY, gridDimZ;
    int blockDimX, blockDimY, blockDimZ;
    int **perSMThreads;

    float alpha, beta;
    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT, matC;

    CUdeviceptr dA, dB, dC;
};
#endif
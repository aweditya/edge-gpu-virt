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
    SGEMMKernel(struct pb_Parameters *params) : params(params)
    {
        alpha = 1.0f;
        beta = 1.0f;
    }
    ~SGEMMKernel() {}

    void memAlloc();
    void memcpyHtoD(const CUstream &stream);
    void memcpyDtoH(const CUstream &stream);
    void memFree();

    void getKernelConfig(unsigned int &gridDimX, unsigned int &gridDimY, unsigned int &gridDimZ,
                         unsigned int &blockDimX, unsigned int &blockDimY, unsigned int &blockDimZ)
    {
        gridDimX = matArow / TILE_M;
        gridDimY = matBcol / TILE_N;
        gridDimZ = 1;

        blockDimX = TILE_N;
        blockDimY = TILE_TB_HEIGHT;
        blockDimZ = 1;
    }

private:
    struct pb_Parameters *params;

    float alpha, beta;
    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT, matC;

    CUdeviceptr dA, dB, dC;
};
#endif
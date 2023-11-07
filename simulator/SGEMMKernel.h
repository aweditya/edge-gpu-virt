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
    SGEMMKernel(pb_Parameters *params) : params(params) {}
    ~SGEMMKernel() {}

    void memAlloc();
    void memcpyHtoD(const CUstream &stream);
    void memcpyDtoH(const CUstream &stream);
    void memFree();

private:
    pb_Parameters *params;

    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT, matC;

    CUdeviceptr dA, dB, dC;
};
#endif
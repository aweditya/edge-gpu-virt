#include "SGEMMKernel.h"

void SGEMMKernel::memAlloc()
{
    matC = std::vector<float>(C_sz);

    checkCudaErrors(cuMemAlloc(&dA, A_sz));
    checkCudaErrors(cuMemAlloc(&dB, B_sz));
    checkCudaErrors(cuMemAlloc(&dC, C_sz));

    args[3] = &blockDimX;
    args[4] = &blockDimY;
    args[5] = &blockDimZ;
    args[6] = perSMThreads;
    args[7] = &dA;
    args[8] = &matArow;
    args[9] = &dB;
    args[10] = &matBcol;
    args[11] = &dC;
    args[12] = &matArow;
    args[13] = &matAcol;
    args[14] = &alpha;
    args[15] = &beta;
}

void SGEMMKernel::memcpyHtoD(const CUstream &stream)
{
    checkCudaErrors(cuMemcpyHtoDAsync(dA, &(matA.front()), A_sz, stream));
    checkCudaErrors(cuMemcpyHtoDAsync(dB, &(matBT.front()), B_sz, stream));
}

void SGEMMKernel::memcpyDtoH(const CUstream &stream)
{
    checkCudaErrors(cuMemcpyDtoHAsync(&(matC.front()), dC, C_sz, stream));
}

void SGEMMKernel::memFree()
{
    checkCudaErrors(cuMemFree(dA));
    checkCudaErrors(cuMemFree(dB));
    checkCudaErrors(cuMemFree(dC));
}
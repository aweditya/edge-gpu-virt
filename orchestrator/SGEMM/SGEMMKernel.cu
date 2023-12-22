#include "SGEMMKernel.h"

void SGEMMKernel::memAlloc()
{
    matC = std::vector<float>(C_sz);

    checkCudaErrors(cuMemAlloc(&dA, A_sz));
    checkCudaErrors(cuMemAlloc(&dB, B_sz));
    checkCudaErrors(cuMemAlloc(&dC, C_sz));

    args[3] = &dA;
    args[4] = &matArow;
    args[5] = &dB;
    args[6] = &matBcol;
    args[7] = &dC;
    args[8] = &matArow;
    args[9] = &matAcol;
    args[10] = &alpha;
    args[11] = &beta;
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
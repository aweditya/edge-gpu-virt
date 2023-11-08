#include "SGEMMKernel.h"

void SGEMMKernel::memAlloc()
{
    readColMajorMatrixFile(params->inpFiles[1], matArow, matAcol, matA);

    readColMajorMatrixFile(params->inpFiles[2], matBcol, matBrow, matBT);

    A_sz = matArow * matAcol * sizeof(float);
    B_sz = matBrow * matBcol * sizeof(float);
    C_sz = matArow * matBcol * sizeof(float);
    matC = std::vector<float>(C_sz);

    checkCudaErrors(cuMemAlloc(&dA, A_sz));
    checkCudaErrors(cuMemAlloc(&dB, B_sz));
    checkCudaErrors(cuMemAlloc(&dC, C_sz));

    args[3] = &dA;
    args[4] = &matArow;
    args[5] = &dB;
    args[6] = &matBcol;
    args[7] = &dC;
    args[8] = &matAcol;
    args[9] = &alpha;
    args[10] = &beta;
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
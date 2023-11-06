#include "sgemm_client.h"
#include "errchk.h"

#include <vector>

void SGEMMClient::setupKernel()
{
    // CUDA memory allocation
    checkCudaErrors(cudaMalloc((void **)&dA, A_sz));
    checkCudaErrors(cudaMalloc((void **)&dB, B_sz));
    checkCudaErrors(cudaMalloc((void **)&dC, C_sz));

    // Copy A and B^T into device memory
    checkCudaErrors(cudaMemcpyAsync(dA, &matA.front(), A_sz, cudaMemcpyHostToDevice, *(kernel.clientStream)));
    checkCudaErrors(cudaMemcpyAsync(dB, &matBT.front(), B_sz, cudaMemcpyHostToDevice, *(kernel.clientStream)));
}

void SGEMMClient::finishKernel()
{
    // Copy C into host memory
    checkCudaErrors(cudaMemcpyAsync(&(matC.front()), dC, C_sz, cudaMemcpyDeviceToHost, *(kernel.clientStream)));

    // CUDA memory deallocation
    checkCudaErrors(cudaFree(dA));
    checkCudaErrors(cudaFree(dB));
    checkCudaErrors(cudaFree(dC));
}
#include "ClockBlockKernel.h"

void ClockBlockKernel::memAlloc()
{
    h_a = (long *)malloc(sizeof(long));
    checkCudaErrors(cuMemAlloc(&d_a, sizeof(long)));

    clock_count = KERNEL_TIME * clockRate;

    args[3] = &blockDimX;
    args[4] = &blockDimY;
    args[5] = &blockDimZ;
    args[6] = perSMThreads;
    args[7] = &d_a;
    args[8] = &clock_count;
}

void ClockBlockKernel::memcpyHtoD(const CUstream &stream)
{
}

void ClockBlockKernel::memcpyDtoH(const CUstream &stream)
{
    checkCudaErrors(cuMemcpyDtoHAsync(h_a, d_a, sizeof(long), stream));
}

void ClockBlockKernel::memFree()
{
    checkCudaErrors(cuMemFree(d_a));

    free(h_a);
}
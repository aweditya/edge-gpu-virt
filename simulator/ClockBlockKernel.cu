#include "ClockBlockKernel.h"

void ClockBlockKernel::memAlloc()
{
    h_a = (long *)malloc(sizeof(long));
    checkCudaErrors(cuMemAlloc(&d_a, sizeof(long)));

    clock_count = 2000L * clockRate;

    args[3] = &d_a;
    args[4] = &clock_count;
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
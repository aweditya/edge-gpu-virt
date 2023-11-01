#include "MatrixAddCallback.h"

void MatrixAddCallback::memAlloc()
{
    h_a = (double *)malloc(N * sizeof(double));
    h_b = (double *)malloc(N * sizeof(double));
    h_c = (double *)malloc(N * sizeof(double));

    checkCudaErrors(cuMemAlloc(&d_a, N * sizeof(double)));
    checkCudaErrors(cuMemAlloc(&d_b, N * sizeof(double)));
    checkCudaErrors(cuMemAlloc(&d_c, N * sizeof(double)));
}

void MatrixAddCallback::memcpyHtoD()
{
    checkCudaErrors(cuMemcpyHtoDAsync(d_a, h_a, N * sizeof(double), stream));
    checkCudaErrors(cuMemcpyHtoDAsync(d_b, h_b, N * sizeof(double), stream));
}

void MatrixAddCallback::memcpyDtoH()
{
    checkCudaErrors(cuMemcpyDtoHAsync(h_c, d_c, N * sizeof(double), stream));
}

void MatrixAddCallback::memFree()
{
    checkCudaErrors(cuMemFree(d_a));
    checkCudaErrors(cuMemFree(d_b));
    checkCudaErrors(cuMemFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);
}
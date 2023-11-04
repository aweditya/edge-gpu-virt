#include "MatrixAddCallback.h"

double drand(const double lo = 0.0, const double hi = 1.0) 
{
    return lo + (hi - lo) / RAND_MAX * rand();
}

void MatrixAddCallback::memAlloc()
{
    h_a = (double *)malloc(N * sizeof(double));
    h_b = (double *)malloc(N * sizeof(double));
    h_c = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i)
    {
        h_a[i] = drand();
        h_b[i] = drand();
    }

    checkCudaErrors(cuMemAlloc(&d_a, N * sizeof(double)));
    checkCudaErrors(cuMemAlloc(&d_b, N * sizeof(double)));
    checkCudaErrors(cuMemAlloc(&d_c, N * sizeof(double)));

    for (int i = 0; i < N; ++i)
    {
        printf("[%d] %.3f, %.3f\n", launcherID, h_a[i], h_b[i]);
    }

    args[3] = &d_a;
    args[4] = &d_b;
    args[5] = &d_c;
}

void MatrixAddCallback::memcpyHtoD(const CUstream &stream)
{
    checkCudaErrors(cuMemcpyHtoDAsync(d_a, h_a, N * sizeof(double), stream));
    checkCudaErrors(cuMemcpyHtoDAsync(d_b, h_b, N * sizeof(double), stream));
}

void MatrixAddCallback::memcpyDtoH(const CUstream &stream)
{
    checkCudaErrors(cuMemcpyDtoHAsync(h_c, d_c, N * sizeof(double), stream));
}

void MatrixAddCallback::memFree()
{
    for (int i = 0; i < N; ++i)
    {
        printf("[%d] %.3f, %.3f, %.3f\n", launcherID, h_a[i], h_b[i], h_c[i]);
    }

    checkCudaErrors(cuMemFree(d_a));
    checkCudaErrors(cuMemFree(d_b));
    checkCudaErrors(cuMemFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);
}
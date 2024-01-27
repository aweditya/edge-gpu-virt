#include "MatrixAddKernel.h"

double drand(const double lo = 0.0, const double hi = 1.0) 
{
    return lo + (hi - lo) / RAND_MAX * rand();
}

void MatrixAddKernel::memAlloc()
{
    h_a = (double *)malloc(M * sizeof(double));
    h_b = (double *)malloc(M * sizeof(double));
    h_c = (double *)malloc(M * sizeof(double));

    for (int i = 0; i < M; ++i)
    {
        h_a[i] = drand();
        h_b[i] = drand();
    }

    checkCudaErrors(cuMemAlloc(&d_a, M * sizeof(double)));
    checkCudaErrors(cuMemAlloc(&d_b, M * sizeof(double)));
    checkCudaErrors(cuMemAlloc(&d_c, M * sizeof(double)));

    // for (int i = 0; i < N; ++i)
    // {
    //     printf("[thread id: %ld] %.3f, %.3f\n", pthread_self(), h_a[i], h_b[i]);
    // }

    args[3] = &d_a;
    args[4] = &d_b;
    args[5] = &d_c;
}

void MatrixAddKernel::memcpyHtoD(const CUstream &stream)
{
    checkCudaErrors(cuMemcpyHtoDAsync(d_a, h_a, M * sizeof(double), stream));
    checkCudaErrors(cuMemcpyHtoDAsync(d_b, h_b, M * sizeof(double), stream));
}

void MatrixAddKernel::memcpyDtoH(const CUstream &stream)
{
    checkCudaErrors(cuMemcpyDtoHAsync(h_c, d_c, M * sizeof(double), stream));
}

void MatrixAddKernel::memFree()
{
    // for (int i = 0; i < N; ++i)
    // {
    //     printf("[thread id: %ld] %.3f, %.3f, %.3f\n", pthread_self(), h_a[i], h_b[i], h_c[i]);
    // }

    checkCudaErrors(cuMemFree(d_a));
    checkCudaErrors(cuMemFree(d_b));
    checkCudaErrors(cuMemFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);
}
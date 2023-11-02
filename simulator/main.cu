#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "KernelLauncher.h"
#include "MatrixAddCallback.h"

CUdevice device;
CUcontext context;
size_t totalGlobalMem;

void initCuda()
{
    int deviceCount = 0;
    checkCudaErrors(cuInit(0));
    int major = 0, minor = 0;

    checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA device
    checkCudaErrors(cuDeviceGet(&device, 0));
    char name[100];
    cuDeviceGetName(name, 100, device);
    printf("> Using device 0: %s\n", name);

    // get compute capabilities and the devicename
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device));
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device));
    printf("  Total amount of global memory:   %llu bytes\n",
           (unsigned long long)totalGlobalMem);
    printf("  64-bit Memory Address:           %s\n",
           (totalGlobalMem > (unsigned long long)4 * 1024 * 1024 * 1024L) ? "YES" : "NO");

    checkCudaErrors(cuCtxCreate(&context, 0, device));
}

void finishCuda()
{
    cuCtxDetach(context);
}

int main(int argc, char **argv)
{
    const std::string moduleFile = "matrixAdd.ptx";
    const std::string kernelName = "_Z8matrxAddPdS_S_i";

    initCuda();

    CUstream stream;
    checkCudaErrors(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    MatrixAddCallback matrixAddCallback;

    kernel_attr_t attr = {.gridDimX = 1,
                          .gridDimY = 1,
                          .gridDimZ = 1,
                          .blockDimX = N,
                          .blockDimY = 1,
                          .blockDimZ = 1,
                          .sharedMemBytes = 0,
                          .stream = stream};
    KernelLauncher launcher(&context, moduleFile, kernelName, &attr, &matrixAddCallback);

    launcher.launch();
    sleep(5);

    checkCudaErrors(cuStreamDestroy(stream));
    finishCuda();

    return 0;
}
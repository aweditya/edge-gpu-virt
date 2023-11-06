#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "KernelWrapper.h"
#include "ClockBlockKernel.h"
#include "FCFSScheduler.h"

CUdevice device;
int clockRate;
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

    // get device properties
    checkCudaErrors(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));

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
    initCuda();
    srand(0);

    FCFSScheduler scheduler;

    const std::string moduleFile1 = "./ptx/clockBlock1.ptx";
    const std::string moduleFile2 = "./ptx/clockBlock2.ptx";
    const std::string kernelName = "clockBlock";

    CUstream stream1, stream2;
    checkCudaErrors(cuStreamCreate(&stream1, CU_STREAM_DEFAULT));
    checkCudaErrors(cuStreamCreate(&stream2, CU_STREAM_DEFAULT));

    ClockBlockKernel clockBlockKernel1(clockRate), clockBlockKernel2(clockRate);
    kernel_attr_t attr1 = {
        .gridDimX = 8,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = 128,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sGridDimX = 8 / 4,
        .sGridDimY = 1,
        .sGridDimZ = 1,
        .sharedMemBytes = 0,
        .stream = stream1};

    kernel_attr_t attr2 = {
        .gridDimX = 8,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = 128,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sGridDimX = 8 / 4,
        .sGridDimY = 1,
        .sGridDimZ = 1,
        .sharedMemBytes = 0,
        .stream = stream2};

    KernelWrapper wrapper1(&scheduler, context, moduleFile1, kernelName, &attr1, &clockBlockKernel1);
    KernelWrapper wrapper2(&scheduler, context, moduleFile2, kernelName, &attr2, &clockBlockKernel2);

    scheduler.run();
    wrapper1.launch();
    wrapper2.launch();

    wrapper1.finish();
    wrapper2.finish();

    scheduler.stop();
    scheduler.finish();

    checkCudaErrors(cuStreamDestroy(stream1));
    checkCudaErrors(cuStreamDestroy(stream2));
    finishCuda();

    return 0;
}

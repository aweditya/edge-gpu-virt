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
    srand(0);

    const std::string moduleFile1 = "./ptx/matrixAdd1.ptx";
    const std::string moduleFile2 = "./ptx/matrixAdd2.ptx";
    const std::string kernelName = "matrixAdd";

    initCuda();

    CUstream stream1, stream2;
    checkCudaErrors(cuStreamCreate(&stream1, CU_STREAM_DEFAULT));
    checkCudaErrors(cuStreamCreate(&stream2, CU_STREAM_DEFAULT));

    MatrixAddCallback matrixAddCallback1, matrixAddCallback2;

    kernel_attr_t attr1 = {
        .gridDimX = N / 128,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = 128,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sGridDimX = N / (128 * 16),
        .sGridDimY = 1,
        .sGridDimZ = 1,
        .sharedMemBytes = 0,
        .stream = stream1};

    kernel_attr_t attr2 = {
        .gridDimX = N / 128,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = 128,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sGridDimX = N / (128 * 16),
        .sGridDimY = 1,
        .sGridDimZ = 1,
        .sharedMemBytes = 0,
        .stream = stream2};

    kernel_control_block_t kcb1, kcb2;
    pthread_mutex_init(&(kcb1.kernel_lock), NULL);
    pthread_cond_init(&(kcb1.kernel_signal), NULL);

    pthread_mutex_init(&(kcb2.kernel_lock), NULL);
    pthread_cond_init(&(kcb2.kernel_signal), NULL);

    KernelLauncher launcher1(rand(), &context, moduleFile1, kernelName, &attr1, &kcb1, &matrixAddCallback1);
    KernelLauncher launcher2(rand(), &context, moduleFile2, kernelName, &attr2, &kcb2, &matrixAddCallback2);

    launcher1.launch();
    launcher2.launch();

    while (true)
    {
        pthread_mutex_lock(&(kcb1.kernel_lock));
        kcb1.slicesToLaunch = 2;
        kcb1.state = RUNNING;
        pthread_cond_signal(&(kcb1.kernel_signal));
        pthread_mutex_unlock(&(kcb1.kernel_lock));

        pthread_mutex_lock(&(kcb2.kernel_lock));
        kcb2.slicesToLaunch = 1;
        kcb2.state = RUNNING;
        pthread_cond_signal(&(kcb2.kernel_signal));
        pthread_mutex_unlock(&(kcb2.kernel_lock));

        if (kcb1.totalSlices == 0 && kcb2.totalSlices == 0)
        {
            break;
        }
    }

    launcher1.finish();
    launcher2.finish();

    checkCudaErrors(cuStreamDestroy(stream1));
    finishCuda();

    return 0;
}

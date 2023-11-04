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
    const std::string kernelName = "matrixAdd";

    initCuda();

    CUstream stream1;
    checkCudaErrors(cuStreamCreate(&stream1, CU_STREAM_DEFAULT));

    MatrixAddCallback matrixAddCallback1;

    kernel_attr_t attr1 = {
        .gridDimX = N,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = 1,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sGridDimX = N / 2,
        .sGridDimY = 1,
        .sGridDimZ = 1,
        .sharedMemBytes = 0,
        .stream = stream1};

    kernel_control_block_t kcb1;
    pthread_mutex_init(&(kcb1.kernel_lock), NULL);
    pthread_cond_init(&(kcb1.kernel_signal), NULL);

    KernelLauncher launcher1(rand(), &context, moduleFile1, kernelName, &attr1, &kcb1, &matrixAddCallback1);

    launcher1.launch();
    pthread_mutex_lock(&(launcher1.kcb->kernel_lock));
    while (launcher1.kcb->state == INIT)
    {
        pthread_cond_wait(&(launcher1.kcb->kernel_signal), &(launcher1.kcb->kernel_lock));
    }
    launcher1.kcb->state = LAUNCH;
    pthread_mutex_unlock(&(launcher1.kcb->kernel_lock));

    while (true)
    {
        launcher1.kcb->slicesToLaunch = 2;
        launcher1.launchKernel();

        if (launcher1.kcb->totalSlices == 0)
        {
            pthread_mutex_lock(&(launcher1.kcb->kernel_lock));
            launcher1.kcb->state = MEMCPYDTOH;
            pthread_cond_signal(&(launcher1.kcb->kernel_signal));
            pthread_mutex_unlock(&(launcher1.kcb->kernel_lock));
            break;
        }
    }

    launcher1.finish();

    pthread_mutex_destroy(&(launcher1.kcb->kernel_lock));
    pthread_cond_destroy(&(launcher1.kcb->kernel_signal));

    checkCudaErrors(cuStreamDestroy(stream1));
    finishCuda();

    return 0;
}

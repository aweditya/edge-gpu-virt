#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <parboil.h>
#include "KernelWrapper.h"
#include "KernelProfiler.h"
#include "SGEMMKernel.h"
#include "RoundRobinScheduler.h"
#include "FCFSScheduler.h"
#include "PriorityScheduler.h"

#define LOGGING_DURATION 10 // Logging duration (in ms)
#define LOGGING_INTERVAL 1 // Logging interval (in us)
#define NUM_KERNELS 2       // Number of kernels to launch

CUdevice device;
int multiprocessorCount;
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
    checkCudaErrors(cuDeviceGetAttribute(&multiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

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
    struct pb_Parameters *params;

    /* Read command line. Expect 3 inputs: A, B^T
       in column-major layout*/
    params = pb_ReadParameters(&argc, argv);
    printf("%s %s %s\n", params->inpFiles[0], params->inpFiles[1], params->inpFiles[2]);
    if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] == NULL))
    {
        fprintf(stderr, "Expecting two input filenames\n");
        exit(-1);
    }

    initCuda();
    srand(0);

    // RoundRobinScheduler scheduler;
    FCFSScheduler scheduler;
    // PriorityScheduler scheduler;

    KernelProfiler profiler(multiprocessorCount, LOGGING_INTERVAL, LOGGING_DURATION);

    const std::string moduleFile = "SGEMM.ptx";
    const std::string kernelName = "SGEMM";

    CUstream streams[NUM_KERNELS];
    std::vector<SGEMMKernel> sgemmKernels;
    sgemmKernels.reserve(NUM_KERNELS);

    kernel_attr_t attrs[NUM_KERNELS];
    std::vector<KernelWrapper> wrappers;
    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        checkCudaErrors(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
        sgemmKernels.emplace_back(params, &profiler.perSMThreads_host);

        sgemmKernels[i].getKernelConfig(attrs[i].gridDimX, attrs[i].gridDimY, attrs[i].gridDimZ,
                                        attrs[i].blockDimX, attrs[i].blockDimY, attrs[i].blockDimZ);

        attrs[i].sGridDimX = attrs[i].gridDimX / 2;
        attrs[i].sGridDimY = attrs[i].gridDimY / 3;
        attrs[i].sGridDimZ = attrs[i].gridDimZ;
        attrs[i].sharedMemBytes = 0;
        attrs[i].stream = streams[i];

        KernelWrapper wrapper(&scheduler, context, moduleFile, kernelName, &attrs[i], &sgemmKernels[i]);
        wrapper.setNiceValue(i % 2);
        wrappers.emplace_back(wrapper);
    }

    struct timeval t0, t1, dt;
    gettimeofday(&t0, NULL);

    scheduler.run();
    profiler.launch();

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        wrappers[i].launch();
    }

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        wrappers[i].finish();
    }

    scheduler.stop();
    scheduler.finish();
    profiler.finish();

    gettimeofday(&t1, NULL);
    timersub(&t1, &t0, &dt);
    printf("[main thread] done in %ld.%06ld\n", dt.tv_sec, dt.tv_usec);

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        checkCudaErrors(cuStreamDestroy(streams[i]));
    }

    finishCuda();

    return 0;
}

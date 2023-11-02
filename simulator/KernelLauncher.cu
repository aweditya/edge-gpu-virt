#include "KernelLauncher.h"

void *KernelLauncher::threadFunction()
{
    checkCudaErrors(cuModuleLoad(&module, moduleFile.c_str()));
    checkCudaErrors(cuModuleGetFunction(&function, module, kernelName.c_str()));
    callback->memAlloc();
    callback->memcpyHtoD(stream);
    launchKernel();
    callback->memcpyDtoH(stream);
    callback->memFree();

    return NULL;
}
#include "KernelLauncher.h"

void *KernelLauncher::threadFunction()
{
    callback->memAlloc();
    callback->memcpyHtoD();
    callback->memcpyDtoH();
    callback->memFree();

    return NULL;
}
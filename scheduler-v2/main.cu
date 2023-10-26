#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <parboil.h>

#include "workload_manager.h"
#include "sgemm_client.h"
#include "errchk.h"

int main(int argc, char **argv)
{
    // Create CUDA stream
    cudaStream_t offeredStream;
    checkCudaErrors(cudaStreamCreate(&offeredStream));

    WorkloadManager manager;

    struct pb_Parameters *params;

    /* Read command line. Expect 3 inputs: A, B and B^T
       in column-major layout*/
    params = pb_ReadParameters(&argc, argv);
    printf("%s %s %s\n", params->inpFiles[0], params->inpFiles[1], params->inpFiles[2]);
    if ((params->inpFiles[0] == nullptr) || (params->inpFiles[1] == nullptr) || (params->inpFiles[2] == nullptr) || (params->inpFiles[3] != nullptr))
    {
        fprintf(stderr, "Expecting three input filenames\n");
        exit(-1);
    }

    SGEMMClient sgemmClient(&manager, &offeredStream, params, 1.0f, 1.0f);
    manager.run();

    // Destroy CUDA stream
    checkCudaErrors(cudaStreamDestroy(offeredStream));

    checkCudaErrors(cudaDeviceReset());

    return 0;
}
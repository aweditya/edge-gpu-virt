#include "clockBlock.h"
#include "KernelProfiler.h"

// This is a kernel that does no real work but runs at least for a specified number of clocks
extern "C" __global__ void clockBlock(int blockOffsetX, int blockOffsetY, int blockOffsetZ,
                                      int blockDimX, int blockDimY, int blockDimZ,
                                      int *perSMThreads,
                                      long *d_o, long clock_count)
{
    int smID;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        smID = get_smid();
        atomicAdd(&perSMThreads[smID], blockDimX * blockDimY * blockDimZ);
    }

    unsigned int start_clock = (unsigned int)clock();

    long clock_offset = 0;

    while (clock_offset < clock_count)
    {
        unsigned int end_clock = (unsigned int)clock();

        // The code below should work like
        // this (thanks to modular arithmetics):
        //
        // clock_offset = (clock_t) (end_clock > start_clock ?
        //                           end_clock - start_clock :
        //                           end_clock + (0xffffffffu - start_clock));
        //
        // Indeed, let m = 2^32 then
        // end - start = end + m - start (mod m).

        clock_offset = (long)(end_clock - start_clock);
    }

    d_o[0] = clock_offset;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        atomicSub(&perSMThreads[smID], blockDimX * blockDimY * blockDimZ);
    }
}

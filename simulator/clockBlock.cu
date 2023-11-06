#include "clockBlock.h"

// This is a kernel that does no real work but runs at least for a specified number of clocks
extern "C" __global__ void clockBlock(int blockOffsetX, int blockOffsetY, int blockOffsetZ, long *d_o, long clock_count)
{
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
}

#ifndef _SM_IDENTIFIER_H
#define _SM_IDENTIFIER_H

#include <cuda.h>
#include <cuda_runtime.h>

// Get ID of SM on which kernel thread is running
__device__ unsigned int get_smid(void)
{
    unsigned int smID;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smID));
    return smID;
}

#endif // _SM_IDENTIFIER
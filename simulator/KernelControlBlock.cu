#include "KernelControlBlock.h"

void kernel_control_block_init(kernel_control_block_t *kcb, unsigned int totalSlices)
{
    pthread_mutex_init(&(kcb->kernel_lock), NULL);
    pthread_cond_init(&(kcb->kernel_signal), NULL);
    kcb->state = INIT;
    kcb->slicesToLaunch = 1;
    kcb->totalSlices = totalSlices;
}

void kernel_control_block_destroy(kernel_control_block *kcb)
{
    pthread_mutex_destroy(&(kcb->kernel_lock));
    pthread_cond_destroy(&(kcb->kernel_signal));
}
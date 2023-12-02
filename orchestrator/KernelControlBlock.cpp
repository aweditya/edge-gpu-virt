#include "KernelControlBlock.h"

void set_state(kernel_control_block_t *kcb, kstate state, bool signal)
{
    pthread_mutex_lock(&(kcb->kernel_lock));
    kcb->state = state;
    if (signal)
    {
        pthread_cond_signal(&(kcb->kernel_signal));
    }
    pthread_mutex_unlock(&(kcb->kernel_lock));
}

void kernel_control_block_init(kernel_control_block_t *kcb, unsigned int totalSlices)
{
    pthread_mutex_init(&(kcb->kernel_lock), NULL);
    pthread_cond_init(&(kcb->kernel_signal), NULL);
    set_state(kcb, INIT);
    kcb->slicesToLaunch = 1;
    kcb->totalSlices = totalSlices;
}

void kernel_control_block_destroy(kernel_control_block *kcb)
{
    pthread_mutex_destroy(&(kcb->kernel_lock));
    pthread_cond_destroy(&(kcb->kernel_signal));
}
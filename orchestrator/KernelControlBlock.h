#ifndef _KERNEL_CONTROL_BLOCK_H
#define _KERNEL_CONTROL_BLOCK_H

#include <pthread.h>

enum kstate
{
    INIT = 0,
    MEMCPYHTOD = 1,
    LAUNCH = 2,
    MEMCPYDTOH = 3
};

typedef struct kernel_control_block
{
    pthread_mutex_t kernel_lock;
    pthread_cond_t kernel_signal;
    kstate state;
    unsigned int slicesToLaunch;
    unsigned int totalSlices;
} kernel_control_block_t;

void set_state(kernel_control_block_t *kcb, kstate state, bool signal = false);
void kernel_control_block_init(kernel_control_block_t *kcb, unsigned int totalSlices);
void kernel_control_block_destroy(kernel_control_block *kcb);

#endif
#pragma once

typedef enum 
{
    READY,
    RUNNING
} KSTATE;

typedef struct kernel_control_block
{
    pthread_mutex_t kernel_lock;
    pthread_cond_t kernel_signal;
    KSTATE state;
    int slices;
} kernel_control_block_t;

void kernel_control_block_init(kernel_control_block_t *kcb)
{
    pthread_mutex_init(&(kcb->kernel_lock), NULL);
    pthread_cond_init(&(kcb->kernel_signal), NULL);
    kcb->state = READY;
    kcb->slices = -1;
}

void kernel_control_block_destroy(kernel_control_block_t *kcb)
{
    pthread_mutex_destroy(&(kcb->kernel_lock));
    pthread_cond_destroy(&(kcb->kernel_signal));
}

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

typedef struct sgemm_args
{
    kernel_control_block_t kcb;

    cudaStream_t stream;
    size_t A_sz, B_sz, C_sz;
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT;

} sgemm_args_t;

typedef struct mriq_args
{
    kernel_control_block_t kcb;

    cudaStream_t stream;
    int numX, numK;      /* Number of X and K values */
    float *kx, *ky, *kz; /* K trajectory (3D vectors) */
    float *x, *y, *z;    /* X coordinates (3D vectors) */
    float *phiR, *phiI;  /* Phi values (complex) */
    float *phiMag;       /* Magnitude of Phi */
    float *Qr, *Qi;      /* Q signal (complex) */

} mriq_args_t;

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

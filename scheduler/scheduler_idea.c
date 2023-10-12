#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct kernel_args
{
    pthread_mutex_t kernel_lock;
    pthread_cond_t kernel_signal;
    int state;
    int slices;
    char name[32];
} kernel_args_t;

void kernel_init(kernel_args_t *kernel, int slices, char *name)
{
    pthread_mutex_init(&(kernel->kernel_lock), NULL);
    pthread_cond_init(&(kernel->kernel_signal), NULL);
    kernel->state = 0;
    kernel->slices = slices;
    memset(kernel->name, 0, 32);
    strcpy(kernel->name, name);
}

void kernel_destroy(kernel_args_t *kernel)
{
    pthread_mutex_destroy(&(kernel->kernel_lock));
    pthread_cond_destroy(&(kernel->kernel_signal));
}

void *kernel_launch(void *args)
{
    kernel_args_t *kernel = (kernel_args_t *)args;

    while (kernel->slices)
    {
        pthread_mutex_lock(&(kernel->kernel_lock));
        while (kernel->state == 0)
            pthread_cond_wait(&(kernel->kernel_signal), &(kernel->kernel_lock));

        printf("Hello! I'm kernel %s\n", kernel->name);
        pthread_mutex_unlock(&(kernel->kernel_lock));

        kernel->state = 0;
        kernel->slices--;
    }
    return NULL;
}

int main()
{
    const int num_threads = 2;
    pthread_t kernel_threads[2];
    kernel_args_t kernels[num_threads];
    char *names[] = {"sgemmNT", "mriq"};

    for (int i = 0; i < num_threads; ++i)
    {
        kernel_init(&(kernels[i]), 5 * (i + 1), names[i]);
    }

    for (int i = 0; i < num_threads; ++i)
    {
        pthread_create(&kernel_threads[i], NULL, kernel_launch, &(kernels[i]));
    }

    int launch = 1;
    while (launch)
    {
        if (launch % 3 == 0)
        {
            pthread_mutex_lock(&(kernels[1].kernel_lock));
            kernels[1].state = 1;
            pthread_cond_signal(&(kernels[1].kernel_signal));
            pthread_mutex_unlock(&(kernels[1].kernel_lock));
        }
        else
        {
            pthread_mutex_lock(&(kernels[0].kernel_lock));
            kernels[0].state = 1;
            pthread_cond_signal(&(kernels[0].kernel_signal));
            pthread_mutex_unlock(&(kernels[0].kernel_lock));
        }
        launch++;
    }

    for (int i = 0; i < num_threads; ++i)
    {
        pthread_join(kernel_threads[i], NULL);
    }

    for (int i = 0; i < num_threads; ++i)
    {
        kernel_destroy(&(kernels[i]));
    }

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct kernel_control_block
{
    pthread_mutex_t kernel_lock;
    pthread_cond_t kernel_signal;
    int state;
    int slices;
    char name[32];
} kernel_control_block_t;

void kernel_control_block_init(kernel_control_block_t *kcb, int slices, char *name)
{
    pthread_mutex_init(&(kcb->kernel_lock), NULL);
    pthread_cond_init(&(kcb->kernel_signal), NULL);
    kcb->state = 0;
    kcb->slices = slices;
    memset(kcb->name, 0, 32);
    strcpy(kcb->name, name);
}

void kernel_control_block_destroy(kernel_control_block_t *kcb)
{
    pthread_mutex_destroy(&(kcb->kernel_lock));
    pthread_cond_destroy(&(kcb->kernel_signal));
}

void *kernel_control_block_launch(void *args)
{
    kernel_control_block_t *kcb = (kernel_control_block_t *)args;

    while (kcb->slices)
    {
        pthread_mutex_lock(&(kcb->kernel_lock));
        while (kcb->state == 0)
            pthread_cond_wait(&(kcb->kernel_signal), &(kcb->kernel_lock));

        printf("Hello! I'm kernel %s\n", kcb->name);
        pthread_mutex_unlock(&(kcb->kernel_lock));

        kcb->state = 0;
        kcb->slices--;
    }
    return NULL;
}

int main()
{
    const int num_threads = 2;
    pthread_t kernel_threads[num_threads];
    kernel_control_block_t kcbs[num_threads];
    char *names[] = {"sgemmNT", "mriq"};

    for (int i = 0; i < num_threads; ++i)
    {
        kernel_control_block_init(&(kcbs[i]), 5 * (i + 1), names[i]);
    }

    for (int i = 0; i < num_threads; ++i)
    {
        pthread_create(&kernel_threads[i], NULL, kernel_control_block_launch, &(kcbs[i]));
    }

    int launch = 1;
    while (launch)
    {
        if (launch % 3 == 0)
        {
            pthread_mutex_lock(&(kcbs[1].kernel_lock));
            kcbs[1].state = 1;
            pthread_cond_signal(&(kcbs[1].kernel_signal));
            pthread_mutex_unlock(&(kcbs[1].kernel_lock));
        }
        else
        {
            pthread_mutex_lock(&(kcbs[0].kernel_lock));
            kcbs[0].state = 1;
            pthread_cond_signal(&(kcbs[0].kernel_signal));
            pthread_mutex_unlock(&(kcbs[0].kernel_lock));
        }

        launch++;

        if (kcbs[0].slices == 0 && kcbs[1].slices == 0)
            launch = 0;
    }

    for (int i = 0; i < num_threads; ++i)
    {
        pthread_join(kernel_threads[i], NULL);
    }

    for (int i = 0; i < num_threads; ++i)
    {
        kernel_control_block_destroy(&(kcbs[i]));
    }

    return 0;
}
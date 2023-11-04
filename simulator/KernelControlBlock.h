#ifndef _KERNEL_CONTROL_BLOCK_H
#define _KERNEL_CONTROL_BLOCK_H

#include "Kernel.h"

void kernel_control_block_init(kernel_control_block_t *kcb, unsigned int totalSlices);
void kernel_control_block_destroy(kernel_control_block *kcb);

#endif
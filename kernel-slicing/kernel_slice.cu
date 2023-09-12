#include<stdio.h>
#include<unistd.h>
#include<cuda.h>
#include<math.h>

/* These are schedular variables */

#define SLICE_SIZE 2   //Number of blocks in each slice


//variables for tracking the progress

unsigned int remaining_blocks=0;
unsigned int blocks_completed=0;
unsigned int total_slices=0;
unsigned int blocks_dispatched=0;


/*   

	KERN_CALL Macro: Should be used by programmer.
        Macro signatues:
XXX         KERN_CALL(kernel_name,total_blocks,threads_per_block,...variables...);  

        currently program sleeps for 2 seconds before dispatching the next slice.

*/


#define KERN_CALL(KERN_NAME,BLOCKS,THREADS,...)  \
        total_slices=ceil((double)(BLOCKS)/(SLICE_SIZE)) ;\
        printf("total_blocks=%d, slice_size=%d, total_slices=%d\n",BLOCKS,SLICE_SIZE,total_slices); \
        remaining_blocks=BLOCKS;\
        for(int slice_index=0;slice_index<total_slices;slice_index++) {\
            if(remaining_blocks>=SLICE_SIZE){\
    	    	blocks_dispatched=SLICE_SIZE;\
	    	    KERN_NAME<<<SLICE_SIZE,THREADS>>>(__VA_ARGS__,slice_index*SLICE_SIZE*THREADS); \
        	}\
        	else if(remaining_blocks<SLICE_SIZE && remaining_blocks!=0){\
		        blocks_dispatched=BLOCKS%SLICE_SIZE;\
        		KERN_NAME<<<remaining_blocks,THREADS>>>(__VA_ARGS__,slice_index*SLICE_SIZE*THREADS); \
        	}\
        	blocks_completed+=blocks_dispatched; \
        	remaining_blocks=BLOCKS-blocks_completed; \
        	printf("\nExecuting slice number %d\n",slice_index+1); \
        	printf(" Blocks Completed=%d, Blocks Remain=%d\n",blocks_completed,remaining_blocks); \
        	cudaDeviceSynchronize(); \
        } 

/*

>cudaDeviceSynchronize() waits for previous slice to complete 
before executing the CPU code


>The sleep() call can be replaced by a shared synchronization variable if multiple
program are trying to launch the CUDA calls.

*/



/*  

    Example kernel function: Taken an integer(int *input)  array and each thread 
    adds its thread-id to corresponding element.  
XXX NOTE that each kernel needs to pass a "threads_offset" variable as the last
    parameter to track the number of blocks executed till now.

*/


__global__ void increment_gpu(int *output, int *input, int total_threads,int thread_offset)
{
	int idx =blockIdx.x*blockDim.x + threadIdx.x +thread_offset; //thread offset needs to added to thread_id
	if(idx<total_threads)
		{
			output[idx]=input[idx]+idx;  //adds the thread id with input value		
		}
}




int main(int argc, char *argv[])
{
	int threads_per_block = 512;  //thread per block
	// int threads_per_block = 16*16;  //thread per block
	int num_blocks=6;  //number of blocks in the program
	// int num_blocks=16*16;  //number of blocks in the program
    
    int total_threads=threads_per_block*num_blocks;
	int total_data = total_threads*sizeof(int);
	int *gpu_input=0, *cpu_input;
	int *gpu_output=0, *cpu_output;

	cpu_input=(int*)malloc(total_threads*sizeof(int));
	cpu_output=(int*)malloc(total_threads*sizeof(int));
	
	for(int i=0;i<total_threads;i++)
		cpu_input[i]=i;  //initialize CPU data
	
	if(!(cudaSuccess==cudaMalloc((void **)&gpu_input, total_data)))  //allocating memory on GPU
		printf("cuda malloc failed!!!\n");
	if(!(cudaSuccess==cudaMalloc((void **)&gpu_output, total_data)))  //allocating memory on GPU
		printf("cuda malloc failed!!!\n");
		
	cudaMemcpy(gpu_input,cpu_input,total_data,cudaMemcpyHostToDevice);  //copying data to GPU

	KERN_CALL(increment_gpu,num_blocks,threads_per_block,gpu_output,gpu_input,total_threads); //Using Macro to call kernel

	cudaMemcpy(cpu_output,gpu_output,total_data,cudaMemcpyDeviceToHost);  //copying back data to CPU

/*

	Checking the results to ensure correctness.
	printing first 10 and last 10 values

*/	
	printf("First 10 values \n");
	for(int i=0;i<10;i++)
		printf("%d\n",cpu_output[i]);
	printf("Last 10 values \n");
	for(int i=total_threads-10;i<total_threads;i++)
		printf("%d\n",cpu_output[i]);
	cudaFree(gpu_input);
	cudaFree(gpu_output);
	
	free(cpu_input);
	free(cpu_output);
	return 0;

}

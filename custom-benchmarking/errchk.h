#ifndef _ERRCHK_H
#define _ERRCHK_H

#include <stdio.h>
#include <cuda_runtime.h>

#define checkCudaErrors(ans) { _checkCudaErrors((ans), __FILE__, __LINE__); }
inline void _checkCudaErrors(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"checkCudaErrors: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif

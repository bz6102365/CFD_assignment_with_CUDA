#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>

static void HandleError(cudaError_t err, const char* file, int line) 
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        system("pause");
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err,__FILE__, __LINE__ ))
#define HANDLE_NULL( a ) { \
    if (a == NULL) { printf( "Host memory failed in %s at line %d\n",\
                __FILE__, __LINE__ ); \
         exit( EXIT_FAILURE );}}

static void CheckCudaKernelStatus(const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n (kernel function exception)", cudaGetErrorString(err), file, line);
        system("pause");
        exit(EXIT_FAILURE);
    }
}
#define CHECK_KERNEL() (CheckCudaKernelStatus(__FILE__, __LINE__ ))

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>
#include <cmath>

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

#define TRUNCTEDLO(x) x>0?x:0
#define TRUNCTEDHI(x,n) x>n?n:x
#define INDEXTRUNCT(x,n) TRUNCTEDLO(TRUNCTEDHI(x,n))

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)


#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

#define minmod(a,b) (a*b>0)*((abs(a)<=abs(b))*a+(abs(a)>abs(b))*b)

#define minmod63(a,b,c,x,y,z) (x<=y)*(x<=z)*a+(y<=x)*(y<=z)*b+(z<=x)*(z<=y)*c

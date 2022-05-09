#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cublas_v2.h>
#include <ctime>

void invert(float** src, float** dst, int n, const int batchSize)
{
    cublasHandle_t handle;
    cublascall(cublasCreate_v2(&handle));

    int* P, * INFO;

    cudacall(cudaMalloc(&P, n * batchSize * sizeof(int)));
    cudacall(cudaMalloc(&INFO, batchSize * sizeof(int)));

    int lda = n;

    float** A = (float**)malloc(batchSize * sizeof(float*));
    float** A_d, * A_dflat;
    cudacall(cudaMalloc(&A_d, batchSize * sizeof(float*)));
    cudacall(cudaMalloc(&A_dflat, n * n * batchSize * sizeof(float)));
    A[0] = A_dflat;

    for (int i = 1; i < batchSize; i++)
        A[i] = A[i - 1] + (n * n);
    cudacall(cudaMemcpy(A_d, A, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    for (int i = 0; i < batchSize; i++)
        cudacall(cudaMemcpy(A_dflat + (i * n * n), src[i], n * n * sizeof(float), cudaMemcpyHostToDevice));

    cublascall(cublasSgetrfBatched(handle, n, A_d, lda, P, INFO, batchSize));

    int* INFOh = new int[batchSize];
    //int INFOh[batchSize];
    cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < batchSize; i++)
        if (INFOh[i] != 0)
        {
            fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }

    float** C = (float**)malloc(batchSize * sizeof(float*));
    float** C_d, * C_dflat;
    cudacall(cudaMalloc(&C_d, batchSize * sizeof(float*)));
    cudacall(cudaMalloc(&C_dflat, n * n * batchSize * sizeof(float)));
    C[0] = C_dflat;
    for (int i = 1; i < batchSize; i++)
        C[i] = C[i - 1] + (n * n);
    cudacall(cudaMemcpy(C_d, C, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    cublascall(cublasSgetriBatched(handle, n, (const float**)A_d, lda, P, C_d, lda, INFO, batchSize));

    cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));


    for (int i = 0; i < batchSize; i++)
        if (INFOh[i] != 0)
        {
            fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    for (int i = 0; i < batchSize; i++)
        cudacall(cudaMemcpy(dst[i], C_dflat + (i * n * n), n * n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(A_d); cudaFree(A_dflat); free(A);
    cudaFree(C_d); cudaFree(C_dflat); free(C);
    cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle); delete[]INFOh;

}

__global__ void matmulKernel(float* dev_points, float* dev_ret, float* dev_invMat, int colunmSize)
{
    dev_ret[threadIdx.x + 1] = 0;
    for (int i = 0; i < colunmSize; i++)
        dev_ret[threadIdx.x + 1] += dev_points[i + 1] * dev_invMat[threadIdx.x * colunmSize + i];
}

__global__ void initKernel(float* dev_points, float* s)
{
    dev_points[1] += 100 * *s;
}

cudaError_t cuSolver_BTCS(OUT float* ret, IN float totalTime, IN float dt, IN float length, IN int n, IN float B)
{
    cudaError_t cudaStatus;

    float cell_length = length / (n - 1);
    float* points = ret;
    points[0] = 100;
    float s = B * dt / (cell_length * cell_length);
    printf("s=%f\n", s);

    float* dev_points = 0;
    float* dev_ret = 0;
    float* dev_s = 0;


    float* dev_coeffMat = 0;

    cudaStatus = cudaMalloc((void**)&dev_s, sizeof(float));
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaMemcpy(dev_s, &s, sizeof(float), cudaMemcpyHostToDevice);
    HANDLE_ERROR(cudaStatus);


    float* coeffMat = 0;

    coeffMat = (float*)malloc((n - 2) * (n - 2) * sizeof(float));

    for (int i = 0; i < n - 2; i++)
    {
        for (int j = 0; j < n - 2; j++)
        {
            coeffMat[i * (n - 2) + j] = 0;
            if (i == j)
                coeffMat[i * (n - 2) + j] = 1 + 2 * s;
            if (abs(i - j) == 1)
                coeffMat[i * (n - 2) + j] = -s;
        }

    }

    float* invMat = 0;
    invMat = (float*)malloc((n - 2) * (n - 2) * sizeof(float));

    float** pCoeffMat = &coeffMat;
    float** pInvMat = &invMat;
    invert(pCoeffMat, pInvMat, (n - 2), 1);
    /*
    for (int i = 0; i < n - 2; i++)
    {
        for (int j = 0; j < n - 2; j++)
        {
            printf("%f ", coeffMat[i * (n - 2) + j]);

        }
        printf("\n");
    }
    */
    float* dev_invMat = 0;

    cudaStatus = cudaMalloc((void**)&dev_invMat, (n - 2) * (n - 2) * sizeof(float));
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaMemcpy(dev_invMat, invMat, (n - 2) * (n - 2) * sizeof(float), cudaMemcpyHostToDevice);
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaMalloc((void**)&dev_points, n * sizeof(float));
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaMemcpy(dev_points, points, n * sizeof(float), cudaMemcpyHostToDevice);
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaMalloc((void**)&dev_ret, n * sizeof(float));
    HANDLE_ERROR(cudaStatus);

    int totalSteps = totalTime / dt;

    float init1 = 100.0f;
    float* dev_init1 = 0;
    cudaStatus = cudaMalloc((void**)&dev_init1, sizeof(float));
    HANDLE_ERROR(cudaStatus);
    cudaStatus = cudaMemcpy((float*)dev_init1, &init1, sizeof(float), cudaMemcpyHostToDevice);
    HANDLE_ERROR(cudaStatus);

    for (int step = 0; step < totalSteps; step++)
    {
        initKernel << <1, 1 >> > (dev_points, dev_s);
        CHECK_KERNEL();

        matmulKernel << <1, n - 2 >> > (dev_points, dev_ret, dev_invMat, n - 2);
        CHECK_KERNEL();

        cudaStatus = cudaMemcpy(dev_points, dev_ret, n * sizeof(float), cudaMemcpyDeviceToDevice);
        HANDLE_ERROR(cudaStatus);

        cudaStatus = cudaMemcpy(dev_points, dev_init1, sizeof(float), cudaMemcpyDeviceToDevice);
        HANDLE_ERROR(cudaStatus);
    }

    cudaStatus = cudaMemcpy(ret, dev_points, n * sizeof(float), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(cudaStatus);

    cudaFree(dev_points);
    cudaFree(dev_ret);
    cudaFree(dev_invMat);
    cudaFree(dev_s);
    cudaFree(dev_init1);
    return cudaStatus;

}

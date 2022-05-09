#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#include <ctime>
clock_t start, end;

cudaError_t cuSolver_FTCS(OUT float* ret, IN float totalTime, IN float dt, IN float length, IN int n, IN float B);

__global__ void solverKernel(float* dev_points, float* dev_ret, float* dev_ATTR);

int main()
{
	int num;
	float* result;

	printf("point num:");
	scanf("%d", &num);

	float dt = 0.0f;
	printf("dt:");
	scanf("%f", &dt);

	float totalTime = 1;
	printf("total time:");
	scanf("%f", &totalTime);

	result = (float*)malloc(num*sizeof(float));
	memset(result, 0, num * sizeof(float));
	result[0] = 100;

	start = clock();

	cuSolver_FTCS(result, totalTime, dt, 1, num, 0.00273);

	end = clock();
	double timeCostOff = (double)(end - start) / CLOCKS_PER_SEC;

	printf("use %.4f sec to caculate\n", timeCostOff);

	system("pause");
	freopen("output.csv", "w", stdout);

	printf("data\n");
	for (int i = 0; i < num; i++)
	{
		printf("%f\n", result[i]);
	}

	fclose(stdout);

	return 0;
}

__global__ void solverKernel(float* dev_points, float* dev_ret, float *dev_s)
{
	int i = threadIdx.x;
	dev_ret[i + 1] = *dev_s * dev_points[i + 2] + (1 - 2 * *dev_s) * dev_points[i + 1] + *dev_s * dev_points[i];
}

cudaError_t cuSolver_FTCS(OUT float* ret, IN float totalTime, IN float dt, IN float length, IN int n, IN float B)
{
	float cell_length = length / (n - 1);
	float* points = ret;
	points[0] = 100;
	float s = B * dt / (cell_length * cell_length);

	float* dev_points=0;
	float* dev_ret = 0;
	float* dev_s = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaMalloc((void**)&dev_points, n * sizeof(float));
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaMemcpy(dev_points, points, n * sizeof(float), cudaMemcpyHostToDevice);
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaMalloc((void**)&dev_ret, n * sizeof(float));
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaMalloc((void**)&dev_s, sizeof(float));
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaMemcpy((float*)dev_s, &s, sizeof(float), cudaMemcpyHostToDevice);
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
		solverKernel <<<1, n - 2 >>> (dev_points, dev_ret, dev_s);
		CHECK_KERNEL();

		cudaStatus = cudaMemcpy(dev_points, dev_ret,n* sizeof(float), cudaMemcpyDeviceToDevice);
		HANDLE_ERROR(cudaStatus);

		cudaStatus = cudaMemcpy(dev_points, dev_init1, sizeof(float), cudaMemcpyDeviceToDevice);
		HANDLE_ERROR(cudaStatus);
	}
	
	cudaStatus = cudaMemcpy(ret, dev_points, n * sizeof(float), cudaMemcpyDeviceToHost);
	HANDLE_ERROR(cudaStatus);
	cudaFree(dev_points);
	cudaFree(dev_ret);
	cudaFree(dev_s);
	cudaFree(dev_init1);

	return cudaStatus;
}
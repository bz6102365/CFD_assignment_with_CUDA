﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cublas_v2.h>
#include <ctime>
#include <iostream>

__global__ void dev_step(float* dev_points, float* dev_ret, float* dev_aPlus, float* dev_aMinus){
	dev_ret[threadIdx.x + 1] = dev_points[threadIdx.x + 1] - *dev_aPlus * (dev_points[threadIdx.x + 1] - dev_points[threadIdx.x]) -*dev_aMinus * (dev_points[threadIdx.x + 2] - dev_points[threadIdx.x + 1]);
}

__global__ void dev_step_1st_and_last(float* dev_points, float* dev_ret, float* dev_aPlus, float* dev_aMinus,int last) {
	dev_ret[0] = dev_points[0] - *dev_aMinus * (dev_points[1] - dev_points[0]);
	dev_ret[last] = dev_points[last] - *dev_aPlus * (dev_points[last] - dev_points[last - 1]);
}

class solver
{
private:
	float* dev_ret = 0;
	float* dev_points = 0;
	float curTime;
	int nums;
	float a;
	float totalTime,dt;
	float* dev_aPlus, * dev_aMinus;

public:
	solver(IN float* initCondition, IN float a, IN int cell_num, IN float totalTime, IN float dt);
	float step();
	void getCurStepData(OUT float* ret);
	~solver();
};

solver::solver(IN float* initCondition, IN float a, IN int cell_num, IN float totalTime, IN float dt)
{
	this->curTime = 0;
	this->nums = cell_num + 1;
	this->totalTime = totalTime;
	this->dt = dt;
	cudaMalloc((void**)&this->dev_points, this->nums * sizeof(float));
	cudaMalloc((void**)&this->dev_ret, this->nums * sizeof(float));
	cudaMalloc((void**)&this->dev_aPlus, sizeof(float));
	cudaMalloc((void**)&this->dev_aMinus, sizeof(float));
	cudaMemcpy(this->dev_points, initCondition, this->nums * sizeof(float), cudaMemcpyHostToDevice);
	float dx = 1.0 / cell_num;
	float aPlus = (a + abs(a)) / 2 * dt / dx;
	float aMinus = (a - abs(a)) / 2 * dt / dx;
	cudaMemcpy(this->dev_aPlus, &aPlus, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->dev_aMinus, &aMinus, sizeof(float), cudaMemcpyHostToDevice);
}

float solver::step()
{
	dev_step_1st_and_last <<< 1, 1 >>> (this->dev_points, this->dev_ret, this->dev_aPlus, this->dev_aMinus,nums-1);
	dev_step <<<1, this->nums-2 >>> (this->dev_points, this->dev_ret, this->dev_aPlus, this->dev_aMinus);
	cudaMemcpy(this->dev_points, this->dev_ret, this->nums * sizeof(float), cudaMemcpyDeviceToDevice);
	CHECK_KERNEL();
	this->curTime += dt;
	return this->curTime;
}

void solver::getCurStepData(OUT float* ret) {
	cudaMemcpy(ret, this->dev_points, this->nums * sizeof(float), cudaMemcpyDeviceToHost);
}

solver::~solver()
{
	cudaFree(this->dev_ret);
	cudaFree(this->dev_points);
	cudaFree(this->dev_aPlus);
	cudaFree(this->dev_aMinus);
}

solver* kernel=0;

extern "C" __declspec(dllexport) void __stdcall initKernel(IN float* initCondition, IN float a, IN int cell_num, IN float totalTime, IN float dt)
{
	kernel = new solver(initCondition, a, cell_num, totalTime, dt);
}

extern "C" __declspec(dllexport) void __stdcall stepKernel()
{
	kernel->step();
}

extern "C" __declspec(dllexport) void __stdcall getData(OUT float* ret)
{
	kernel->getCurStepData(ret);
}

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		std::cout << "LINK SHOKAN!" << std::endl;
		break;

	case DLL_PROCESS_DETACH:
		//std::cout << "DLL_PROCESS_DETACH" << std::endl;
		delete kernel;
		break;
	}

	return TRUE;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cublas_v2.h>
#include <ctime>
#include <iostream>

__global__ void dev_FVS(float* dev_points, float* dev_positive_ret, float* dev_negative_ret,float a) {
	dev_positive_ret[threadIdx.x] = 0.5 * a * (abs(dev_points[threadIdx.x]) + dev_points[threadIdx.x]);
	dev_negative_ret[threadIdx.x] = 0.5 * a * (-abs(dev_points[threadIdx.x]) + dev_points[threadIdx.x]);
}

__global__ void dev_calc_f_plus_half_plus(float* dev_f_plus, float* dev_ret) {
	float q0 = 1.0 / 3.0 * dev_f_plus[threadIdx.x] - 7.0 / 6.0 * dev_f_plus[threadIdx.x + 1] + 11.0 / 6.0 * dev_f_plus[threadIdx.x + 2];
	float q1 = -1.0 / 6.0 * dev_f_plus[threadIdx.x + 1] + 5.0 / 6.0 * dev_f_plus[threadIdx.x + 2] + 1.0 / 3.0 * dev_f_plus[threadIdx.x + 3];
	float q2 = 1.0 / 3.0 * dev_f_plus[threadIdx.x + 2] + 5.0 / 6.0 * dev_f_plus[threadIdx.x + 3] - 1.0 / 6.0 * dev_f_plus[threadIdx.x + 4];

	float is0 = 13.0 / 12.0 * (dev_f_plus[threadIdx.x] - 2.0 * dev_f_plus[threadIdx.x + 1] + dev_f_plus[threadIdx.x + 2]) * \
		(dev_f_plus[threadIdx.x] - 2 * dev_f_plus[threadIdx.x + 1] + dev_f_plus[threadIdx.x + 2])\
		+ 1.0 / 4.0 * (dev_f_plus[threadIdx.x] - 4.0 * dev_f_plus[threadIdx.x + 1] + 3.0 * dev_f_plus[threadIdx.x + 2])\
		* (dev_f_plus[threadIdx.x] - 4.0 * dev_f_plus[threadIdx.x + 1] + 3.0 * dev_f_plus[threadIdx.x + 2]);

	float is1 = 13.0 / 12.0 * (dev_f_plus[threadIdx.x + 1] - 2.0 * dev_f_plus[threadIdx.x + 2] + dev_f_plus[threadIdx.x + 3]) * \
		(dev_f_plus[threadIdx.x + 1] - 2 * dev_f_plus[threadIdx.x + 2] + dev_f_plus[threadIdx.x + 3])\
		+ 1.0 / 4.0 * (dev_f_plus[threadIdx.x + 1] - dev_f_plus[threadIdx.x + 3])\
		* (dev_f_plus[threadIdx.x + 1] - dev_f_plus[threadIdx.x + 3]);

	float is2 = 13.0 / 12.0 * (dev_f_plus[threadIdx.x + 2] - 2.0 * dev_f_plus[threadIdx.x + 3] + dev_f_plus[threadIdx.x + 4]) * \
		(dev_f_plus[threadIdx.x + 2] - 2.0 * dev_f_plus[threadIdx.x + 3] + dev_f_plus[threadIdx.x + 4])\
		+ 1.0 / 4.0 * (3.0 * dev_f_plus[threadIdx.x + 2] - 4.0 * dev_f_plus[threadIdx.x + 3] + 1.0 * dev_f_plus[threadIdx.x + 4])\
		* (3.0 * dev_f_plus[threadIdx.x + 2] - 4.0 * dev_f_plus[threadIdx.x + 3] + 1.0 * dev_f_plus[threadIdx.x + 4]);

	dev_ret[threadIdx.x + 2] = minmod63(q0, q1, q2, is0, is1, is2);
}

__global__ void dev_calc_f_plus_half_minus(float* dev_f_plus, float* dev_ret) {
	float q0 = -1.0 / 6.0 * dev_f_plus[threadIdx.x] +5.0 / 6.0 * dev_f_plus[threadIdx.x + 1] + 1.0 / 3.0 * dev_f_plus[threadIdx.x + 2];
	float q1 = 1.0 / 3.0 * dev_f_plus[threadIdx.x + 1] + 5.0 / 6.0 * dev_f_plus[threadIdx.x + 2] - 1.0 / 6.0 * dev_f_plus[threadIdx.x + 3];
	float q2 = 11.0 / 6.0 * dev_f_plus[threadIdx.x + 2] - 7.0 / 6.0 * dev_f_plus[threadIdx.x + 3] + 1.0 / 3.0 * dev_f_plus[threadIdx.x + 4];

	float is0 = 13.0 / 12.0 * (dev_f_plus[threadIdx.x] - 2.0 * dev_f_plus[threadIdx.x + 1] + dev_f_plus[threadIdx.x + 2]) * \
		(dev_f_plus[threadIdx.x] - 2 * dev_f_plus[threadIdx.x + 1] + dev_f_plus[threadIdx.x + 2])\
		+ 1.0 / 4.0 * (dev_f_plus[threadIdx.x] - 4.0 * dev_f_plus[threadIdx.x + 1] + 3.0 * dev_f_plus[threadIdx.x + 2])\
		* (dev_f_plus[threadIdx.x] - 4.0 * dev_f_plus[threadIdx.x + 1] + 3.0 * dev_f_plus[threadIdx.x + 2]);

	float is1 = 13.0 / 12.0 * (dev_f_plus[threadIdx.x + 1] - 2.0 * dev_f_plus[threadIdx.x + 2] + dev_f_plus[threadIdx.x + 3]) * \
		(dev_f_plus[threadIdx.x + 1] - 2 * dev_f_plus[threadIdx.x + 2] + dev_f_plus[threadIdx.x + 3])\
		+ 1.0 / 4.0 * (dev_f_plus[threadIdx.x + 1] - dev_f_plus[threadIdx.x + 3])\
		* (dev_f_plus[threadIdx.x + 1] - dev_f_plus[threadIdx.x + 3]);

	float is2 = 13.0 / 12.0 * (dev_f_plus[threadIdx.x + 2] - 2.0 * dev_f_plus[threadIdx.x + 3] + dev_f_plus[threadIdx.x + 4]) * \
		(dev_f_plus[threadIdx.x + 2] - 2.0 * dev_f_plus[threadIdx.x + 3] + dev_f_plus[threadIdx.x + 4])\
		+ 1.0 / 4.0 * (3.0 * dev_f_plus[threadIdx.x + 2] - 4.0 * dev_f_plus[threadIdx.x + 3] + 1.0 * dev_f_plus[threadIdx.x + 4])\
		* (3.0 * dev_f_plus[threadIdx.x + 2] - 4.0 * dev_f_plus[threadIdx.x + 3] + 1.0 * dev_f_plus[threadIdx.x + 4]);

	dev_ret[threadIdx.x + 1] = minmod63(q0, q1, q2, is0, is1, is2);
}

__global__ void dev_f_reduce(float* dev_f_plus, float* dev_f_minus, float* dev_ret) {
	dev_ret[threadIdx.x] = dev_f_plus[threadIdx.x] + dev_f_minus[threadIdx.x];
}

__global__ void dev_non_rk_diff(float* dev_f, float* dev_ret, float dt,float dx) {
	dev_ret[threadIdx.x + 3] = (dev_f[threadIdx.x + 3] - dev_f[threadIdx.x + 2]) / dx * dt;
}

__global__ void dev_fill_diff(float* dev_points, float* dev_ret, float dt, float dx,int last) {
	dev_ret[0] = 0;
	dev_ret[1] = (dev_points[1] - dev_points[0]) * dt / dx;
	dev_ret[2] = (dev_points[2] - dev_points[1]) * dt / dx;
	dev_ret[last] = (dev_points[last] - dev_points[last - 1]) * dt / dx;
	dev_ret[last - 1] = (dev_points[last - 1] - dev_points[last - 2]) * dt / dx;
	dev_ret[last - 2] = (dev_points[last - 2] - dev_points[last - 3]) * dt / dx;
}


__global__ void dev_non_rk_step(float* dev_points,float* dev_diff, float* dev_ret) {
	dev_ret[threadIdx.x] = dev_points[threadIdx.x] - dev_diff[threadIdx.x];
}


__global__ void dev_rk3_reduce(float* dev_points, float* dev_ret, float* dev_diff0, float* dev_diff1, float* dev_diff2) {
	dev_ret[threadIdx.x] = dev_points[threadIdx.x] - 1.0 / 6.0 * (dev_diff0[threadIdx.x] + 4 * dev_diff1[threadIdx.x] + dev_diff2[threadIdx.x]);
}


void step_without_rk(IN float* dev_points, OUT float* dev_ret, float dt, float nums, float a) 
{
	float* dev_f_plus = 0;
	float* dev_f_minus = 0;
	cudaMalloc((void**)&dev_f_plus, nums * sizeof(float));
	cudaMalloc((void**)&dev_f_minus, nums * sizeof(float));

	dev_FVS <<<1, nums >>> (dev_points, dev_f_plus, dev_f_minus, a);

	float* dev_f_plus_half = 0;
	float* dev_f_minus_half = 0;
	cudaMalloc((void**)&dev_f_plus_half, nums * sizeof(float));
	cudaMalloc((void**)&dev_f_minus_half, nums * sizeof(float));

	dev_calc_f_plus_half_plus <<<1, nums - 4 >>> (dev_f_plus, dev_f_plus_half);
	dev_calc_f_plus_half_plus <<<1, nums - 4 >>> (dev_f_plus, dev_f_plus_half);

	float* dev_f_reduced = 0;
	cudaMalloc((void**)&dev_f_reduced, nums * sizeof(float));

	dev_f_reduce <<<1, nums >>> (dev_f_plus, dev_f_minus, dev_f_reduced);

	float* dev_diff = 0;
	cudaMalloc((void**)&dev_diff, nums * sizeof(float));

	dev_non_rk_diff<<<1, nums - 6 >>>(dev_f_reduced, dev_diff, dt, 1.0 / (nums - 1));

	dev_fill_diff <<<1, 1 >>> (dev_points, dev_diff, dt, 1.0 / (nums - 1), nums - 1);

	dev_non_rk_step <<<1, nums >>> (dev_points, dev_diff, dev_ret);

	cudaMemcpy(dev_points, dev_ret, nums * sizeof(float), cudaMemcpyDeviceToDevice);

	cudaFree(dev_f_plus);
	cudaFree(dev_f_minus);
	cudaFree(dev_f_plus_half);
	cudaFree(dev_f_minus_half);
	cudaFree(dev_f_reduced);
	cudaFree(dev_diff);

	CHECK_KERNEL();
}

void single_step_diff(IN float* dev_points, OUT float* dev_ret, float dt, float nums, float a) //k*dt
{
	float* dev_f_plus = 0;
	float* dev_f_minus = 0;
	cudaMalloc((void**)&dev_f_plus, nums * sizeof(float));
	cudaMalloc((void**)&dev_f_minus, nums * sizeof(float));

	dev_FVS <<<1, nums >>> (dev_points, dev_f_plus, dev_f_minus, a);

	float* dev_f_plus_half = 0;
	float* dev_f_minus_half = 0;
	cudaMalloc((void**)&dev_f_plus_half, nums * sizeof(float));
	cudaMalloc((void**)&dev_f_minus_half, nums * sizeof(float));

	dev_calc_f_plus_half_plus <<<1, nums - 4 >>> (dev_f_plus, dev_f_plus_half);
	dev_calc_f_plus_half_plus <<<1, nums - 4 >>> (dev_f_plus, dev_f_plus_half);

	float* dev_f_reduced = 0;
	cudaMalloc((void**)&dev_f_reduced, nums * sizeof(float));

	dev_f_reduce <<<1, nums >>> (dev_f_plus, dev_f_minus, dev_f_reduced);

	float* dev_diff = 0;
	cudaMalloc((void**)&dev_diff, nums * sizeof(float));

	dev_non_rk_diff <<<1, nums - 6 >>> (dev_f_reduced, dev_diff, dt, 1.0 / (nums - 1));

	dev_fill_diff <<<1, 1 >>> (dev_points, dev_diff, dt, 1.0 / (nums - 1), nums - 1);

	cudaMemcpy(dev_ret, dev_diff, nums * sizeof(float), cudaMemcpyDeviceToDevice);

	cudaFree(dev_f_plus);
	cudaFree(dev_f_minus);
	cudaFree(dev_f_plus_half);
	cudaFree(dev_f_minus_half);
	cudaFree(dev_f_reduced);
	cudaFree(dev_diff);

	CHECK_KERNEL();
}

__global__ void dev_coeff_step(float* dev_points, float* dev_diff, float* dev_ret, float coeff) {
	dev_ret[threadIdx.x] = dev_points[threadIdx.x] - coeff * dev_diff[threadIdx.x];
}


void single_step_rk3(IN float* dev_points, OUT float* dev_ret, float dt, float nums, float a)
{
	float* dev_diff1 = 0;
	cudaMalloc((void**)&dev_diff1, nums * sizeof(float));

	single_step_diff(dev_points, dev_diff1, dt, nums, a);

	float* dev_points2 = 0;
	cudaMalloc((void**)&dev_points2, nums * sizeof(float));
	
	dev_coeff_step <<<1, nums >>> (dev_points, dev_diff1, dev_points2, 0.5f);
	//////////////////////////////////////////
	float* dev_diff2 = 0;
	cudaMalloc((void**)&dev_diff2, nums * sizeof(float));

	single_step_diff(dev_points2, dev_diff2, dt, nums, a);

	float* dev_points3 = 0;
	cudaMalloc((void**)&dev_points3, nums * sizeof(float));

	dev_coeff_step <<<1, nums >>> (dev_points, dev_diff1, dev_points3, -1.0);
	dev_coeff_step <<<1, nums >>> (dev_points3, dev_diff2, dev_points3, 2.0);
	//////////////////////////////////////////
	float* dev_diff3 = 0;
	cudaMalloc((void**)&dev_diff3, nums * sizeof(float));
	single_step_diff(dev_points3, dev_diff3, dt, nums, a);

	dev_rk3_reduce <<<1, nums >>> (dev_points, dev_ret, dev_diff1, dev_diff2, dev_diff3);

	cudaMemcpy(dev_points, dev_ret, nums * sizeof(float), cudaMemcpyDeviceToDevice);

	cudaFree(dev_diff1);
	cudaFree(dev_diff2);
	cudaFree(dev_diff3);
	cudaFree(dev_points3);
	cudaFree(dev_points2);
}

class solver
{
private:
	float* dev_ret = 0;
	float* dev_points = 0;
	float curTime;
	int nums;
	float a;
	float totalTime, dt;
	float* dev_s;

public:
	solver(IN float* initCondition, IN float a, IN int cell_num, IN float totalTime, IN float dt);
	float step();
	//void rk_step();
	void getCurStepData(OUT float* ret);
	~solver();
};

solver::solver(IN float* initCondition, IN float a, IN int cell_num, IN float totalTime, IN float dt)
{
	this->a = a;
	this->curTime = 0;
	this->nums = cell_num + 1;
	this->totalTime = totalTime;
	this->dt = dt;
	cudaMalloc((void**)&this->dev_points, this->nums * sizeof(float));
	cudaMalloc((void**)&this->dev_ret, this->nums * sizeof(float));
	cudaMalloc((void**)&this->dev_s, sizeof(float));
	cudaMemcpy(this->dev_points, initCondition, this->nums * sizeof(float), cudaMemcpyHostToDevice);
	float dx = 1.0 / cell_num;
	float s = a * dt / dx;
	cudaMemcpy(this->dev_s, &s, sizeof(float), cudaMemcpyHostToDevice);
}

float solver::step()
{
	//step_without_rk(this->dev_points, this->dev_ret, this->dt, this->nums, this->a);
	single_step_rk3(this->dev_points, this->dev_ret, this->dt, this->nums, this->a);
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
	cudaFree(this->dev_s);
}

solver* kernel = 0;

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

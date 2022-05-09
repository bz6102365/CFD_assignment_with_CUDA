#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "solver.h"
#include <stdio.h>

int main()
{
	float* result61, * result121, * result241;//61 121 241

	result61 = (float*)malloc(61 * sizeof(float));
	result121 = (float*)malloc(121 * sizeof(float));
	result241 = (float*)malloc(241 * sizeof(float));

	cuSolver_BTCS(result61, 100.0f, 0.0001, 1.0f, 61, 0.00273);
	cuSolver_BTCS(result121, 100.0f, 0.0001, 1.0f, 121, 0.00273);
	cuSolver_BTCS(result241, 100.0f, 0.0001, 1.0f, 241, 0.00273);

	float* resBlk60, * resBlk120, * resBlk240;
	resBlk60 = (float*)malloc(240 * sizeof(float));
	resBlk120 = (float*)malloc(240 * sizeof(float));
	resBlk240 = (float*)malloc(240 * sizeof(float));

	for (int i = 0; i < 60; i++) //
	{
		resBlk60[4 * i + 1] = (result61[i] + result61[i + 1]) / 2;
		resBlk60[4 * i + 2] = (result61[i] + result61[i + 1]) / 2;
		resBlk60[4 * i + 3] = (result61[i] + result61[i + 1]) / 2;
		resBlk60[4 * i] = (result61[i] + result61[i + 1]) / 2;
	}

	for (int i = 0; i < 120; i++) //
	{
		resBlk120[2 * i + 1] = (result121[i] + result121[i + 1]) / 2;
		resBlk120[2 * i] = (result121[i] + result121[i + 1]) / 2;
	}

	for (int i = 0; i < 240; i++) //
	{
		resBlk240[i] = (result241[i] + result241[i + 1]) / 2;
	}

	float e21 = 0;
	float e32 = 0;
	float* p= (float*)malloc(240 * sizeof(float));
	float pTotal = 0;
	for (int i = 0; i < 240; i++)
	{
		e32 = resBlk240[i] - resBlk120[i];
		e21 = resBlk120[i] - resBlk60[i];
		p[i] = log(abs(e32/e21))/log(0.5);
		printf("p[%d]=%f\n", i, p[i]);
		pTotal += p[i];
	}
	pTotal = pTotal / 240;
	printf("p=%f", pTotal);
	return 0;
}
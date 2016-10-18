/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Demonstration of inline PTX (assembly language) usage in CUDA kernels
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#define STREAM 4
#define INTER 1000000
__global__ void fma_fp16(int a, int b, int c, int* res)
{
	int a1=a;
	int b1=b;
	int c1=c;

	int a2=a+res[0];
	int b2=b+res[0];
	int c2=c+res[0];
	int a3=a+res[1];
	int b3=b+res[1];
	int c3=c+res[1];
	int a4=a+res[2];
	int b4=b+res[2];
	int c4=c+res[2];

	int elemID = blockIdx.x * blockDim.x + threadIdx.x;

	int i;
	for(i=0;i<INTER;i++)
	{
		asm volatile ("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(c1) : "r"(a1), "r"(b1), "r"(c1));
		asm volatile ("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(c2) : "r"(a2), "r"(b2), "r"(c2));
		asm volatile ("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(c3) : "r"(a3), "r"(b3), "r"(c3));
		asm volatile ("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(c4) : "r"(a4), "r"(b4), "r"(c4));
	}
	res[STREAM*elemID]=c1;
	res[STREAM*elemID+1]=c2;
	res[STREAM*elemID+2]=c3;
	res[STREAM*elemID+3]=c4;
}

__global__ void fma_fp32(int a, int b, int c, int* res)
{
	int a1=a;
	int b1=b;
	int c1=c;

	int a2=a+res[0];
	int b2=b+res[0];
	int c2=c+res[0];
	int a3=a+res[1];
	int b3=b+res[1];
	int c3=c+res[1];
	int a4=a+res[2];
	int b4=b+res[2];
	int c4=c+res[2];

	int elemID = blockIdx.x * blockDim.x + threadIdx.x;

	int i;
	for(i=0;i<INTER;i++)
	{
		asm volatile ("fma.rn.f32 %0, %1, %2, %3;" : "=r"(c1) : "r"(a1), "r"(b1), "r"(c1));
		asm volatile ("fma.rn.f32 %0, %1, %2, %3;" : "=r"(c2) : "r"(a2), "r"(b2), "r"(c2));
		asm volatile ("fma.rn.f32 %0, %1, %2, %3;" : "=r"(c3) : "r"(a3), "r"(b3), "r"(c3));
		asm volatile ("fma.rn.f32 %0, %1, %2, %3;" : "=r"(c4) : "r"(a4), "r"(b4), "r"(c4));
	}
	res[STREAM*elemID]=c1;
	res[STREAM*elemID+1]=c2;
	res[STREAM*elemID+2]=c3;
	res[STREAM*elemID+3]=c4;
}
 
__global__ void fma_int8(int a, int b, int c, int* res)
{
	int a1=a;
	int b1=b;
	int c1=c;
	int a2=a+res[0];
	int b2=b+res[0];
	int c2=c+res[0];
	int a3=a+res[1];
	int b3=b+res[1];
	int c3=c+res[1];
	int a4=a+res[2];
	int b4=b+res[2];
	int c4=c+res[2];

	int elemID = blockIdx.x * blockDim.x + threadIdx.x;

	int i;
	for(i=0;i<INTER;i++)
	{
		asm volatile ("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(c1) : "r"(a1), "r"(b1), "r"(c1));
		asm volatile ("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(c2) : "r"(a2), "r"(b2), "r"(c2));
		asm volatile ("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(c3) : "r"(a3), "r"(b3), "r"(c3));
		asm volatile ("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(c4) : "r"(a4), "r"(b4), "r"(c4));
	}
	res[STREAM*elemID]=c1;
	res[STREAM*elemID+1]=c2;
	res[STREAM*elemID+2]=c3;
	res[STREAM*elemID+3]=c4;
}


int main(int argc, char **argv)
{
	printf("Theoretical computing power tests\n");
	int threadNum=512;
	int blockNum=400;
	const int N = STREAM*blockNum*threadNum;

        cudaEvent_t start, stop;
	int dev = findCudaDevice(argc, (const char **) argv);

	if (dev == -1)
	{
		return EXIT_FAILURE;
	}
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));


	int *d_ptr;
	checkCudaErrors(cudaMalloc(&d_ptr, N * sizeof(int)));

	int *h_ptr;
	checkCudaErrors(cudaMallocHost(&h_ptr, N * sizeof(int)));

	dim3 cudaBlockSize(threadNum,1,1);
	dim3 cudaGridSize(blockNum, 1, 1);

	checkCudaErrors(cudaEventRecord(start, NULL));
	int round=1;
	int i;
	for(i=0;i<round;i++)
	{
		fma_fp16<<<cudaGridSize, cudaBlockSize>>>(3,4,5,d_ptr);
	} 
	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, N *sizeof(int), cudaMemcpyDeviceToHost));
	printf("FP16 done \n");
	float msecTotal = 0.0f;
	double ops=(double)STREAM*(double)INTER*(double)threadNum*(double)blockNum*4*round;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	double opsps=(ops*1.0e-9f)/(msecTotal/1000.0f);
	printf(
			"Performance= %.2fG FP16 op/s, Time= %.3f msec, Size= %.0f Ops\n",
			opsps,
			msecTotal,
			ops);

	checkCudaErrors(cudaEventRecord(start, NULL));
	for(i=0;i<round;i++)
	{
		fma_int8<<<cudaGridSize, cudaBlockSize>>>(3,4,5,d_ptr);
	} 
	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, N *sizeof(int), cudaMemcpyDeviceToHost));
	printf("int8 done \n");
	ops=(double)STREAM*(double)INTER*(double)threadNum*(double)blockNum*8*round;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	opsps=(ops*1.0e-9f)/(msecTotal/1000.0f);
	printf(
			"Performance= %.2fG INT8 op/s, Time= %.3f msec, Size= %.0f Ops\n",
			opsps,
			msecTotal,
			ops);

	checkCudaErrors(cudaEventRecord(start, NULL));
	for(i=0;i<round;i++)
	{
		fma_fp32<<<cudaGridSize, cudaBlockSize>>>(3,4,5,d_ptr);
	} 
	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, N *sizeof(int), cudaMemcpyDeviceToHost));
	printf("fp32 done \n");
	ops=(double)STREAM*(double)INTER*(double)threadNum*(double)blockNum*2*round;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	opsps=(ops*1.0e-9f)/(msecTotal/1000.0f);
	printf(
			"Performance= %.2fG FP32 op/s, Time= %.3f msec, Size= %.0f Ops\n",
			opsps,
			msecTotal,
			ops);



	checkCudaErrors(cudaFree(d_ptr));
	checkCudaErrors(cudaFreeHost(h_ptr));


	// Calling cudaProfilerStop causes all profile data to be
	// flushed before the application exits
	checkCudaErrors(cudaProfilerStop());

	return 0;
}

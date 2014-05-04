/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "catsparser.cuh"
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {					\
		cudaError_t _m_cudaStat = value;			\
		if (_m_cudaStat != cudaSuccess) {			\
			fprintf(stderr, "Error %d %s at line %d in file %s\n", \
					value,cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
					exit(1);					\
		} }

__device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}
template<int _rows,int _collumns>

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

struct stats {
	int collumns;
	int start;
};

__device__ int __forceinline__ bland(float * cost, int n) 
{
	const unsigned int laneid = threadIdx.x%32;

	float vmax = 0.0f;
	int imin = -1;
	unsigned int status = 0;

	for(int i = laneid; i < n; i += 32) {
		float t = cost[i];
		vmax = fmin(vmax,t);
		status = __ballot(vmax < 0.0f);		
		if(status) {
			if(laneid == 0) {
				imin = i + __ffs(status) - 1;		
			}
			break;			
		}		
	}
	imin = __shfl(imin,0);

	return imin;	
}



template<int rows>
__device__ int __forceinline__ get_pivot_row(float  ( *p)[rows],int collumn, int collumns)
{
	const unsigned int laneid = threadIdx.x%32;
	float vmax = INFINITY;
	int index = -1;

	const float div = p[collumn][laneid];
	const float top = p[collumns-1][laneid];

	if(div > 0.0f) {
		vmax = top/div;
	}
	const float t = vmax;
	vmax = fmin(vmax,__shfl_xor(vmax,16));
	vmax = fmin(vmax,__shfl_xor(vmax, 8));
	vmax = fmin(vmax,__shfl_xor(vmax, 4));
	vmax = fmin(vmax,__shfl_xor(vmax, 2));
	vmax = fmin(vmax,__shfl_xor(vmax, 1));
	//	printf("id %d Ratio %f min %f\n",laneid,t,vmax);
	index = __ballot(vmax == t);
	if(laneid == 0) {
		index = __ffs(index) - 1;
	}
	index = __shfl(index,0);
	return index;
}
template<int rows>
__device__ int __forceinline__ is_integer(float  (*p)[rows],int collumns) {
	const unsigned int laneid = threadIdx.x%32;
	float in = p[collumns-1][laneid];
	int result = __all(ceilf(in) == in);
	return result;
}

template<int rows>
__device__ void pivot(float  (*p)[rows],float * cost, int collumns, int collumn, int row) {
	const unsigned int laneid = threadIdx.x%32;

	/*
	 *  // everything but row p and column q
	 *  for (int i = 0; i <= M; i++)
	 *     for (int j = 0; j <= M + N; j++)
	 *        if (i != p && j != q) a[i][j] -= a[p][j] * a[i][q] / a[p][q];
	 * 	i = row, j = collumn
	 *  a[p][q] = pivot element
	 *  a[i][q] = row element in collumn q
	 *
	 *  0 = p*x+y
	 *  -y = p*x
	 *  -y/p = x
	 *
	 */
	const float ratio = p[collumn][laneid]/p[collumn][row];
	//__shared__ float tmp[32][4][32];

	//tmp[warpid][laneid]

	for(int c = 0; c < collumns; c ++) {
		if(c == collumn) continue;
		//float res =
		float put = p[c][laneid];
		if(laneid == row) {
			put /= p[collumn][row];
			cost[c] -= put*cost[collumn];
		} else {
			put -= p[c][row]*(ratio);
		}
		p[c][laneid] =put;

	}

	p[collumn][laneid] = (float) (laneid == row);
	if(laneid == row) {
		cost[collumn] = 0.0f;
	}

}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
//only works for 32 rows!!
template<int rows>
__device__ float __forceinline__ apply_row_op(float  (*p)[rows],float * cost, int collumns)
{
	//__shared__ int bids[32][32];
	//bids[warpid][laneid] = -1;
	//assert(cost[0] == -2.0f);
	while(1) {
		int row, collumn;
		collumn = bland(cost,collumns);
		if(collumn == -1 || cost[collumn] >= 0.0f) {
			break;
		}
		row = get_pivot_row<rows>(p,collumn,collumns);
		//printf("lane %d col %d row %d cost %f\n",laneid,collumn,row,cost[collumn]);
	//	if(row == -1) {
		//	count = 2;
		//}
		//		if(laneid == 0) {
		//			printf("warpid %d col %d row %d %f\n",warpid,collumn,row,p[collumn][row]);
		//		}
	//	assert(row >= 0);
		pivot<rows>(p,cost,collumns,collumn,row);
		__threadfence();
		//bids[warpid][]
		//__syncthreads();
		//		break;
		continue;
	}
	return cost[collumns-1];
}

template<int rows>
__device__ int __forceinline__  init_table(float  (*matrix)[rows],float * cost,int n, unsigned int * in, unsigned int * in_value) {
	const unsigned int tid = threadIdx.x;
	const unsigned int laneid = tid % 32;
	const unsigned int mask =  1 << laneid;
	int i;
	for(i = 0; i < n; i ++) {
		matrix[i][laneid] =(float) ((in[i]  & mask) > 0);
	}
	//now i == n
	for(;i<n+32;i++) {
		matrix[i][laneid] = ((float)((i-n) == laneid));
	}
	//now i == n+32, put p collumn
	matrix[i][laneid] = 0.0f;
	i++;
	//put constraint
	matrix[i][laneid] = 1.0f;

	int c;
	for(c = laneid; c< n; c +=32) {
		cost[c] = -((float) in_value[c]);
	}
	for(;c < n+32; c += 32) {
		cost[c] = 0.0f;
	}
	// now c = n+32
	if(laneid == 0) {
		cost[n+32] = 1.0f;
		cost[n+32+1] = 0.0f;
	}
	return i+1;
}


template<int rows>
__global__ void do_simplex(float  (*matrix)[rows],float * cost,int n,unsigned int * in, unsigned int * in_value) {


	const unsigned int warpid = (threadIdx.x+ blockIdx.x * blockDim.x) / 32;

	//__shared__ int cache2[1024];
	const int new_n = n + 32 + 2;
	init_table<rows>((matrix+warpid*new_n),(cost+warpid*new_n),n,in,in_value);
	//	__syncthreads();

	//	float  (*p)[rows] = (matrix+warpid*new_n);
	//	float  (*p2)[rows]= (matrix+(warpid+1)*new_n);
	//	if(warpid < 31) {
	//		for(int i = 0; i < new_n;i++) {
	//			assert(p[i][laneid] == p2[i][laneid]);
	//		}
	//	}
	//
	//	return;
	apply_row_op<rows>((matrix+warpid*new_n),(cost+warpid*new_n),new_n);
	//	__threadfence_system();
	//__syncthreads();
	//assert(cache2[tid] == cache2[(tid+32)%blockDim.x]);
	//assert(collumn == cache2[tid]);
	//	printf("t1 %d i %d, warp %d\n",cache2[tid],tid,warpid);

	//assert(cache2[i] == cache2[tid]);
	//assert(cache[laneid][laneid] == cache[warpid][laneid]);

}

#define set(X) (1<<(X))
#define set2(X,Y) (set(X)|set(Y))
int main(int argc, const char* argv[]) {
	int warps = (1024*14)/32;
	const int problemwidth = 21;
	const int rows = 32;
	const int collumns = problemwidth+2+rows;
	float matrix[collumns][rows];
	unsigned int bids[problemwidth] = {set(5),set2(4,5),set2(2,4),set2(2,3),set(3),set2(1,3),
			46554,88465,122321,4654848,6545645,1321321,3215484,64555,
			665565,12324,32132,32122,548484,989498,456542};
	unsigned int value [problemwidth] = {2,3,4,6,8,1,14,58,64,21,32,45,65,1,23,45,84,65,32,12,45};
	float cost[collumns];
	//+1 for the constraints
	int i;
	for(i = 0; i < problemwidth;i++) {
		for(int j = 0; j < rows;j++) {
			matrix[i][j] = 0.0 + ((float) !!(set(j) & bids[i]));

		}
		cost[i] =-((float) value[i]);
		//printf("%f\n",cost[i]);
	}
	//return 0;
	//put the I matrix in mem
	for(; i < problemwidth+rows;i++) {
		for(int j =0; j < rows;j++) {
			matrix[i][j] = 0.0f;
			if(j == i-problemwidth) {
				matrix[i][j] = 1.0f;
			}
		}
	}

	//set the constraints
	for(int j=0; j < rows;j++) {
		matrix[problemwidth+rows][j] = 0.0f;
		matrix[problemwidth+rows+1][j] = 1.0f;
	}

	for(i = 0; i < rows; i ++ ) {
		for(int c = 0; c < collumns; c++) {
			printf("%.1f\t",matrix[c][i]);
		}
		printf("\n");
	}
	//set rest of cost vector to 0
	for(int c = problemwidth; c < collumns; c++) {
		cost[c] = 0.0f;
	}
	//set p to 1
	cost[collumns -2] = 1.0f;

	for(int c = 0; c < collumns; c++) {
		printf("%.1f\t",cost[c]);
	}

	float (* dmatrix)[32];
	CUDA_CHECK_RETURN(cudaMalloc((void**) &dmatrix,sizeof(matrix)*warps));
	printf("allocated %lu bytes for matrix\n",sizeof(matrix)*warps);
	float * mmatrix = (float *)  dmatrix;
	float * dcost;
	unsigned int * in_value;
	unsigned int * in;
	CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &in,sizeof(unsigned int)*problemwidth));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &in_value,sizeof(unsigned int)*problemwidth));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &dcost,sizeof(cost)*warps));
	printf("allocated %lu bytes for cost\n",sizeof(cost)*warps);

	CUDA_CHECK_RETURN(cudaMemcpy(in,bids,sizeof(unsigned int)*problemwidth, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(in_value,value,sizeof(unsigned int)*problemwidth, cudaMemcpyHostToDevice));
	//return 0;
	//	for(int w = 0; w < warps; w++) {
	//	CUDA_CHECK_RETURN(cudaMemcpy(&mmatrix[w*(sizeof(matrix)/sizeof(float))],matrix,sizeof(matrix), cudaMemcpyHostToDevice));
	//	CUDA_CHECK_RETURN(cudaMemcpy(&dcost[w*(sizeof(cost)/sizeof(float))],cost,sizeof(cost), cudaMemcpyHostToDevice));
	//}

	//CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(float) * 26843545));
	printf("hello\n");
	do_simplex<32><<<7,1024>>>(dmatrix,dcost,problemwidth,in,in_value);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	//return 0;
	CUDA_CHECK_RETURN(cudaMemcpy(matrix,mmatrix,sizeof(matrix), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(cost,dcost,sizeof(cost), cudaMemcpyDeviceToHost));
	for(i = 0; i < rows; i ++ ) {
		for(int c = 0; c < collumns; c++) {
			printf("%.1f\t",matrix[c][i]);
		}
		printf("\n");
	}
	for(int c = 0; c < collumns; c++) {
		printf("%.1f\t",cost[c]);
	}
	exit(0);
	if(argc < 2) {
		fprintf(stderr,"No argument supplied\n");
		exit(EXIT_FAILURE);
	}
	for(int i = 0; i < argc; i++) {
		printf("Argument %d : %s\n",i,argv[i]);
	}

	struct config * config = parse_file(argv[1]);
	//	void *d = NULL;
	//	int i;
	//	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];
	//
	//	for (i = 0; i < WORK_SIZE; i++)
	//		idata[i] = (unsigned int) i;
	//
	//	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE));
	//	CUDA_CHECK_RETURN(
	//			cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));
	//
	//	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);
	//
	//	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	//	CUDA_CHECK_RETURN(cudaGetLastError());
	//	CUDA_CHECK_RETURN(cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));
	//
	//	for (i = 0; i < WORK_SIZE; i++)
	//		printf("Input value: %u, device output: %u\n", idata[i], odata[i]);
	//
	//	CUDA_CHECK_RETURN(cudaFree((void*) d));
	//	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}

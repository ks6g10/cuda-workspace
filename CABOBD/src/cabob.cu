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
__device__ float __forceinline__ apply_row_op(float  (*p)[rows],float * cost, int collumns,unsigned short * basis)
{
	//__shared__ int bids[32][32];
	//bids[warpid][laneid] = -1;
	//assert(cost[0] == -2.0f);
	//(collumns-rows-2) = n bids
	basis[threadIdx.x%32] = (collumns-rows-2) + (threadIdx.x%32);
	while(1) {
		int row, collumn;
		collumn = bland(cost,collumns);
		if(collumn == -1 || cost[collumn] >= 0.0f) {
			break;
		}
		row = get_pivot_row<rows>(p,collumn,collumns);
		if(threadIdx.x%32 == 0) {
			basis[collumn] = row;
		}
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

template<int size>
struct stack {
	unsigned int variables[32];
	unsigned int confs[size];
	unsigned int value[size];
};

enum estack{
	VALUE = 0,
	CONF = 1,
	SIZE = 2,
	OFFSET = 3
};

enum rstatus {
	COMPLETE = -1,
	NOEDGE = -2
};


template<int rows>
__device__ int __forceinline__  init_table(float  (*matrix)[rows],float * cost,int n, unsigned int * in, unsigned int * in_value) {
	const unsigned int tid = threadIdx.x;
	const unsigned int laneid = tid % 32;
	const unsigned int mask =  1 << laneid;
	int count = 0;
	int i;
	for(i = 0; i < n; i ++) {
		count += ((in[i]  & mask) > 0);
		matrix[i][laneid] =(float) ((in[i]  & mask) > 0);
	}
	if(__ballot(count == n)) {
		return COMPLETE;//COMPLETE
	} else if (!__ballot(count > 1)) {
		return NOEDGE;//NO EDGES
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


template<int rows, int stack_size,int stacks_per_warp>
__device__ int add_initial_bids(struct stack<stack_size> * mystack, int index, int bids, unsigned int * in, unsigned int * in_value, unsigned int laneid) {
	int size = 0;
	const unsigned int root = in[index];
	const unsigned int ltmask = (1 << laneid) - 1;
	for(int i = laneid+index+1; i < bids; i += 32) {
		unsigned int conf = in[i];
		unsigned int status = (conf & root);
		unsigned int status2;
		if((status2 = __ballot(status == 0))) {
			int put = __popc((status2 & ltmask));
			if(status == 0) {
				mystack->confs[put+size] = conf;
				mystack->value[put+size] = in_value[i];
			}
			size += __shfl(put+(status == 0),31);
		}
	}
	if(laneid == 0) {
		mystack->variables[VALUE] = in_value[index];
		mystack->variables[CONF] = root;
		mystack->variables[SIZE] = size;
		mystack->variables[OFFSET] = 0;
	}
	return size;

}

template<int rows, int stack_size,int stacks_per_warp>
__device__ int __forceinline__ add_bids_new_stack(struct stack<stack_size> * parent,struct stack<stack_size> * child,int index,unsigned int laneid, unsigned int backtrack) {

	int size = 0;

	if(parent->variables[SIZE] == 0) {
		return 0;
	}
	if(child->variables[OFFSET] == 0 && backtrack) {
		return -1;
	}


	const unsigned int root = parent->confs[index] | parent->variables[CONF];
	const unsigned int ltmask = (1 << laneid) - 1;
	if(laneid == 0) {
		parent->variables[SIZE] -=1;

		child->variables[VALUE] = parent->value[index] + parent->variables[VALUE];
	}

	//remove the added bid from prev graph
	for(int i = index+laneid+1; i < parent->variables[SIZE]+1; i +=32) {
		unsigned int conf = parent->confs[i];
		unsigned int value = parent->value[i];
		parent->confs[i-1] = conf;
		parent->value[i-1] = value;
	}



	for(int i = laneid; i < parent->variables[SIZE]; i += 32) {
		unsigned int conf = parent->confs[i];
		unsigned int status = (conf & root);
		unsigned int status2;
		if((status2 = __ballot(status == 0))) {
			int put = __popc((status2 & ltmask));
			if(status == 0) {
				child->confs[put+size] = conf;
				child->value[put+size] = parent->value[i];
			}
			size += __shfl(put+(status == 0),31);
		}
	}



	if(laneid == 0) {

		child->variables[CONF] = root;
		child->variables[SIZE] = size;
		child->variables[OFFSET] = parent->variables[OFFSET]+1;
	}


	return size;
}

template<int rows, int stack_size,int stacks_per_warp>
__device__ void __forceinline__ recursion(struct stack<stack_size> * stack,unsigned int * global_max,float  (*matrix)[rows],float * cost, unsigned short * basis,const int re) {
	const unsigned int laneid = threadIdx.x%32;
	if(re >= rows) {
		return;
	}
	if(stack->variables[SIZE] == 0) {
		return;
	}
	int status = init_table<rows>(matrix,cost,stack->variables[SIZE],stack->confs,stack->value);

	if(status == COMPLETE) {
		unsigned int tmax = 0;
		for(int i = laneid;i < stack->variables[SIZE];i +=32) {
			tmax = max(tmax,stack->value[i]);
		}

		tmax = max(tmax,(unsigned int)__shfl_xor((int)tmax,16));
		tmax = max(tmax,(unsigned int)__shfl_xor((int)tmax, 8));
		tmax = max(tmax,(unsigned int)__shfl_xor((int)tmax, 4));
		tmax = max(tmax,(unsigned int)__shfl_xor((int)tmax, 2));
		tmax = max(tmax,(unsigned int)__shfl_xor((int)tmax, 1));

		if(laneid == 0) {
			atomicMax(global_max,tmax+stack->variables[VALUE]);
		}
		return;
	} else if(status == NOEDGE) {
		unsigned int tmax = 0;
		for(int i = laneid;i < stack->variables[SIZE];i +=32) {
			tmax += stack->value[i];
		}

		tmax += (unsigned int)__shfl_xor((int)tmax,16);
		tmax += (unsigned int)__shfl_xor((int)tmax,8);
		tmax += (unsigned int)__shfl_xor((int)tmax,4);
		tmax += (unsigned int)__shfl_xor((int)tmax,2);
		tmax += (unsigned int)__shfl_xor((int)tmax,1);

		if(laneid == 0) {
			atomicMax(global_max,tmax+stack->variables[VALUE]);
		}
		return;
	}

	unsigned int estimate = apply_row_op<rows>(matrix,cost,stack->variables[SIZE]+32 + 2,basis);
	estimate += stack->variables[VALUE];
	if(estimate < *global_max) {
		return;
	} else if(is_integer<rows>(matrix,stack->variables[SIZE] + 32 + 2)) {
		atomicMax(global_max,estimate);
		return;
	}
	while(stack->variables[SIZE]) {
		int size= add_bids_new_stack<rows,stack_size,stacks_per_warp>(stack,(stack+1),0,laneid,0);
		if(size == 0) continue;

		recursion<rows,stack_size,stacks_per_warp>((stack+1),global_max,matrix,cost,basis,re+1);
	}
	return;
}

#define MOFFSET (gwarpid*(n + 32 + 2))
template<int rows, int stack_size,int stacks_per_warp>
__global__ void do_simplex(struct stack<stack_size> * stacks,unsigned int * global_max, unsigned int * atom, float  (*matrix)[rows],float * cost,int n,unsigned int * in, unsigned int * in_value) {


	const unsigned int gwarpid = (threadIdx.x+ blockIdx.x * blockDim.x) / 32;
	const unsigned int warpid = threadIdx.x/32;
	const unsigned int laneid = threadIdx.x%32;
	__shared__ struct stack<stack_size> * stack[rows];
	__shared__ unsigned short basis[32][rows];
	if(laneid == 0) {
		stack[warpid] = (stacks+gwarpid*stacks_per_warp);
	}
	newalloc:
	int index = -1;
	if(laneid == 0) {
		index = atomicAdd(atom,1);
	}
	index = __shfl(index,0);
	if(index >= n) {
		//if(laneid == 0)
		//printf("%u\n",global_max);
		return;
	}
	if(add_initial_bids<rows,stack_size,stacks_per_warp>(stack[warpid],index,n,in,in_value,laneid) == 0) {
		goto newalloc;
	}

	recursion<rows,stack_size,stacks_per_warp>(stack[warpid],global_max,(matrix+MOFFSET),(cost+MOFFSET),basis[warpid],0);

	goto newalloc;

}

#define set(X) (1<<(X))
#define set2(X,Y) (set(X)|set(Y))
int main(int argc, const char* argv[]) {
	//int warps = (1024*14)/32;
	const int problemwidth = 21;
	const int rows = 32;
	const int collumns = problemwidth+2+rows;
	const int blocks = 7;
	const int threads = 1024;
	const int warps = blocks*(threads/32);

	const int stack_width = rows;
	const int stacks_per_warp = rows;
	const int stacks = warps*stacks_per_warp;


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
	unsigned int * dmax;//
	unsigned int * datom;
	unsigned int * in_value;
	unsigned int * in;
	struct stack<stack_width> * dstacks;

	CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &dmax,sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &datom,sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &dstacks,sizeof(struct stack<stack_width>)*stacks));

	CUDA_CHECK_RETURN(cudaMalloc((void**) &in,sizeof(unsigned int)*problemwidth));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &in_value,sizeof(unsigned int)*problemwidth));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &dcost,sizeof(cost)*warps));
	printf("allocated %lu bytes for cost\n",sizeof(cost)*warps);
	CUDA_CHECK_RETURN(cudaMemset(dmax,0,sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemset(datom,0,sizeof(unsigned int)));

	CUDA_CHECK_RETURN(cudaMemcpy(in,bids,sizeof(unsigned int)*problemwidth, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(in_value,value,sizeof(unsigned int)*problemwidth, cudaMemcpyHostToDevice));
	//return 0;
	//	for(int w = 0; w < warps; w++) {
	//	CUDA_CHECK_RETURN(cudaMemcpy(&mmatrix[w*(sizeof(matrix)/sizeof(float))],matrix,sizeof(matrix), cudaMemcpyHostToDevice));
	//	CUDA_CHECK_RETURN(cudaMemcpy(&dcost[w*(sizeof(cost)/sizeof(float))],cost,sizeof(cost), cudaMemcpyHostToDevice));
	//}

	//CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(float) * 26843545));
	printf("hello\n");
	do_simplex<rows,stack_width,stacks_per_warp><<<7,1024>>>(dstacks,dmax,datom,dmatrix,dcost,problemwidth,in,in_value);
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

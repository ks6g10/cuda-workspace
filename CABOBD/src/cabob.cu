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
			fprintf(stderr, "Error %s at line %d in file %s\n", \
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
					exit(1);					\
		} }

__device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}
template<int _rows,int _collumns>
struct frame {
	const int rows;
	const int collumns;
	float matrix[_collumns][_rows];
	float cost[_collumns];
};


/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

//make sure cost is aligned to 32-256 overspend and nulled to 0.0f
#define DEFAULT_FLOAT (0.0f)
template<int lim>
__device__ int __forceinline__ find_entering_var_old(float * cost, int n) {
	const unsigned int tid = threadIdx.x;
	const unsigned int warpid = tid/32;
	const unsigned int laneid = tid%32;

	float vmax = 0.0f;
	int imax = laneid;
	/*
	 * cache[0]. x y z w
	 *           0 1 2 3
	 * cache[1]. x y z w
	 *           128 129 130 131
	 */
	if(lim <= 32) {
		if(laneid < n) {
			vmax = cost[laneid];
		}
	} else if(lim <= 64) {
		float2 cache = make_float2(DEFAULT_FLOAT,DEFAULT_FLOAT);
		imax = laneid*2;
		if(laneid*2 < n) {
			cache = ((float2 *) cost)[laneid];
			cache.y *= (laneid*2+1 < n);

		}
		vmax = fmax(cache.x,cache.y);
		if(cache.y == vmax) {
			imax++;
		}
	} else if (lim <= 128) {
		float4 cache = make_float4(DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT);

		if(laneid*4 < n) {
			cache = ((float4 *) cost)[laneid];
			cache.y *= (laneid*4+1 < n);
			cache.z *= (laneid*4+2 < n);
			cache.w *= (laneid*4+3 < n);
		}
		cache.x = fmax(cache.x,cache.y);
		cache.z = fmax(cache.z,cache.w);
		vmax = fmax(cache.x,cache.z);
		//set base index to x
		imax = laneid*4;
		if (vmax == cache.y) {
			imax +=1;
		} else if(vmax == cache.z) {
			imax += 2;
		} else if (vmax == cache.w) {
			imax += 3;
		}
	} else if ( lim <= 256) {

		float4 cache[2];
		cache[0] = ((float4 *) cost)[laneid];//guaranteed to be in mem, as n > 128
		cache[1] = ((float4 *) cost)[laneid+32];
		if(laneid*4+4+4*32 >= n) {
			// ignore constant 4*32 in calc
			/* laneid 0: n 1: 0+4-1= 3: yzw
			 * laneid 0: n 2: 0+4-2= 2: zw
			 * laneid 0: n 3: 0+4-3= 1: w
			 * laneid 0: n 4: 0+4-4= 0:
			 * laneid 4: n 4: 16+4-4= 16: xyzw
			 * laneid 3: n 4: 3+4-4= 3: yzw
			 */
			switch(laneid*4+4+4*32 -n) {
			case 3:
				cache[1] = make_float4(cache[0].x,DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT);
				break;
			case 2:
				cache[1] = make_float4(cache[0].x,cache[0].y,DEFAULT_FLOAT,DEFAULT_FLOAT);
				break;
			case 1:
				cache[1].w = DEFAULT_FLOAT;
				break;
			default:
				cache[1] = make_float4(DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT);
				break;
			}
		}

		cache[0].x = fmax(cache[0].x,cache[0].y);
		cache[1].x = fmax(cache[1].x,cache[1].y);

		cache[0].z = fmax(cache[0].z,cache[0].w);
		cache[1].z = fmax(cache[1].z,cache[1].w);

		cache[0].x = fmax(cache[0].x,cache[0].z);
		cache[1].x = fmax(cache[1].x,cache[1].z);

		vmax = fmax(vmax,cache[0].x);
		vmax = fmax(vmax,cache[1].x);
		if(vmax == cache[0].x) {
			imax = laneid*4;
			if(vmax == cache[0].y) {
				imax += 1;
			} else if(vmax == cache[0].z) {
				imax += 2;
			} else if(vmax == cache[0].w) {
				imax += 3;
			}

		} else if(vmax == cache[1].x){
			//4*32== the offset of the first half of the fetched number
			// as each thread in the warp fetches a float4
			imax = laneid*4+4*32;
			if(vmax == cache[1].y) {
				imax += 1;
			} else if(vmax == cache[1].z) {
				imax += 2;
			} else if(vmax == cache[1].w) {
				imax += 3;
			}
		}
	} else {
		/*256->512->768->1024
		 *
		 *
		 */
		for(int i = 0; i < n/32*8; i+=2) {
			float4 cache[2];
			cache[0] = ((float4 *) cost)[laneid+32*4*i];
			cache[1] = ((float4 *) cost)[laneid+32*4*(i+1)];
			cache[0].x = fmax(cache[0].x,cache[0].y);
			cache[1].x = fmax(cache[1].x,cache[1].y);

			cache[0].z = fmax(cache[0].z,cache[0].w);
			cache[1].z = fmax(cache[1].z,cache[1].w);

			cache[0].x = fmax(cache[0].x,cache[0].z);
			cache[1].x = fmax(cache[1].x,cache[1].z);
			vmax = fmax(vmax,cache[0].x);
			vmax = fmax(vmax,cache[1].x);

			int index0 = -1;
			index0 += (vmax == cache[0].x);
			index0 += (vmax == cache[0].z)*2;
			index0 += (vmax == cache[0].w);
			index0 += !(vmax == cache[0].z)*(vmax == cache[0].y);
		}
	}
	float t = vmax;
	for (int mask = warpSize/2; mask > 0; mask /= 2)
		vmax = fmax(vmax,__shfl_xor(vmax, mask));

	imax = __ffs(__ballot(vmax == t))-1;

	return imax;
}
#define DEFAULT_FLOAT (0.0f)
template<int lim>
__device__ int __forceinline__ find_entering_var(float * cost, int n) {
	const unsigned int tid = threadIdx.x;
	const unsigned int laneid = tid%32;

	float vmax = DEFAULT_FLOAT;
	int imax = -1;
	/*
	 * cache[0]. x y z w
	 *           0 1 2 3
	 * cache[1]. x y z w
	 *           128 129 130 131
	 */
	if(lim <= 32) {
		if(laneid < n) {
			vmax = cost[laneid];
			if(vmax < 0.0f) {
				imax = laneid;
			}

		}
	} else if(lim <= 64) {
		float2 cache = make_float2(DEFAULT_FLOAT,DEFAULT_FLOAT);
		imax = laneid*2;
		if(laneid*2 < n) {
			cache = ((float2 *) cost)[laneid];
			cache.y *= (laneid*2+1 < n);

		}
		vmax = fmin(cache.x,cache.y);
		if(cache.y == vmax) {
			imax++;
		}
	} else if (lim <= 128) {
		float4 cache = make_float4(DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT);

		if(laneid*4 < n) {
			cache = ((float4 *) cost)[laneid];
			cache.y *= (laneid*4+1 < n);
			cache.z *= (laneid*4+2 < n);
			cache.w *= (laneid*4+3 < n);
		}
		cache.x = fmin(cache.x,cache.y);
		cache.z = fmin(cache.z,cache.w);
		vmax = fmin(cache.x,cache.z);
		//set base index to x
		imax = laneid*4;
		if (vmax == cache.y) {
			imax +=1;
		} else if(vmax == cache.z) {
			imax += 2;
		} else if (vmax == cache.w) {
			imax += 3;
		}
	} else {
		float4 cache = make_float4(DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT);
		float4 cache2 = make_float4(DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT);
		int i;
#pragma unroll 2
		for(i = laneid; i*4+4 < n; i+=32) {

			/* i 0: n 1: 0+4-1= 3: yzw
			 * i 0: n 2: 0+4-2= 2: zw
			 * i 0: n 3: 0+4-3= 1: w
			 * i 0: n 4: 0+4-4= 0:
			 * i 1: n 5: 4+4-5= 3: yzw
			 * i 4: n 4: 16+4-4= 16: xyzw
			 * i 3: n 4: 3+4-4= 3: yzw
			 */
			cache = ((float4 *) cost)[i];
			const int tindex = i*4;
			cache.x = fmin(cache.x,cache.y);
			cache.z = fmin(cache.z,cache.w);
			float tmax = fmin(cache.x,cache.z);
			if(tmax < vmax) {
				vmax = tmax;
				imax = (tmax == cache.x)*(tindex);
				imax = (tmax == cache.y)*(tindex+1);
				imax = (tmax == cache.z)*(tindex+2);
				imax = (tmax == cache.w)*(tindex+3);
			}

		}

		if(i*4 < n) {
			cache = ((float4 *) cost)[i];
			switch(i*4+4 -n) {
			case 3:
				cache = make_float4(cache.x,DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT);
				break;
			case 2:
				cache = make_float4(cache.x,cache.y,DEFAULT_FLOAT,DEFAULT_FLOAT);
				break;
			case 1:
				cache.w = DEFAULT_FLOAT;
				break;
			default:
				cache = make_float4(DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT,DEFAULT_FLOAT);
				break;
			}
			const int tindex = i*4;
			cache.x = fmin(cache.x,cache.y);
			cache.z = fmin(cache.z,cache.w);
			vmax = fmin(cache.x,cache.z);
			if (vmax == cache.y) {
				imax =tindex+1;
			} else if (vmax == cache.w) {
				imax =tindex+3;
			} else if(vmax == cache.z) {
				imax =tindex+2;
			} else if (vmax == cache.x) {
				imax =tindex;
			}
		}

	}

	float t = vmax;
	for (int mask = warpSize/2; mask > 0; mask /= 2)
		vmax = fmin(vmax,__shfl_xor(vmax, mask));

	imax = __shfl(imax,__ffs(__ballot(vmax == t))-1);

	return imax;
}


template<int rows>
__device__ int __forceinline__ get_pivot_row(float  p[][rows],int collumn, int collumns)
{
	const unsigned int laneid = threadIdx.x%32;
	float vmax = INFINITY;
	int index = -1;

	for(int i = laneid; i < rows; i +=32) {
		float div = p[collumn][i];
		float top = p[collumns-1][i];
		if(div > 0.0f) {
			float frac = top/div;
			if(frac < vmax) {
				vmax = frac;
				index = i;
			}
		}
	}
	float t = vmax;
	vmax = fmin(vmax,__shfl_xor(vmax,16));
	vmax = fmin(vmax,__shfl_xor(vmax, 8));
	vmax = fmin(vmax,__shfl_xor(vmax, 4));
	vmax = fmin(vmax,__shfl_xor(vmax, 2));
	vmax = fmin(vmax,__shfl_xor(vmax, 1));
	index = __shfl(index,__ffs(__ballot(vmax == t))-1);
	return index;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
//only works for 32 rows!!
template<int rows, int lim>
__device__ float __forceinline__ apply_row_op(float  p[][rows],float * cost, int collumns)
{

	//
	const unsigned int laneid = threadIdx.x%32;
	const unsigned int warpid = threadIdx.x/32;

	int row, collumn;
	while(1) {
		collumn = find_entering_var<lim>(cost,collumns);
		if(collumn == -1 || cost[collumn] >= 0.0f) {
			break;						
		}

		row = get_pivot_row<rows>(p,collumn,collumns);



		float colpiv = (*p)[warpid][collumn][laneid];
		float div = __shfl(colpiv,row);
		div /=div;
		if(laneid == row) {
			div = colpiv;
			colpiv /= colpiv;
		}

		float y = (-colpiv)/div;
		// 0 = xdiv + colpiv
		// -colpiv = xdiv
		// -colpiv/div = x
		for(int c = 0; c < collumn; c++) {
			float element = (*p)[warpid][i][laneid];
			if(laneid == row) {
				element /=div;
			}
			float xpp = __shfl(element,row);
			if(laneid != row) {
				element= y*xpp+element;
			}
			(*p)[warpid][i][laneid] = element;
		}

		for(int c = collumn+1; c < collumns; c++) {
			float element = (*p)[i][laneid];
			if(laneid == row) {
				element /=div;
			}
			float xpp = __shfl(element,row);
			if(laneid != row) {
				element= y*xpp+element;
			}
			(*p)[i][laneid] = element;
		}
	}

	return cost[collumns-1]
}


template<int rows,int collumns>
__global__ void do_simplex(float  matrix[][collumns][rows],float * cost,int n) {


	const unsigned int tid = threadIdx.x;
	int cache2[1024];
	//	cache2[tid] = 0;

	//	if(tid < n) {
	//		cost[tid] = cosf(1/tid+0.5555555555555555555f);
	//	}
	__syncthreads();
	int collumn = find_entering_var<26843545>(cost,n);
	if(cost[collumn] > 0.0f) {

	}
	get_pivot_row<32>(matrix,collumn,n);

	cache2[tid] = collumn;
	__threadfence_system();
	__syncthreads();
	assert(collumn == cache2[tid]);
	for(int i = 0; i < 1024;i++) {
		//printf("t1 %d t2 %d, i %d, laneid %d\n",collumn,cache[i][laneid],i,laneid);
		assert(collumn == cache2[tid]);
	}
	//assert(cache[laneid][laneid] == cache[warpid][laneid]);

}

#define set(X) (1<<(X-1))
#define set2(X,Y) (set(X)|set(Y))
int main(int argc, const char* argv[]) {
	float * d = NULL;
	int warps = 1024*14/32;
	const int problemwidth = 6;
	const int rows = 32;
	struct frame<rows,problemwidth+1+rows> prob ={rows,problemwidth+1};
	unsigned int bids[problemwidth] = {set(5),set2(4,5),set2(2,4),set2(2,3),set(3),set2(1,3)};
	float value [problemwidth] = {2.f,3.f,4.f,6.f,8.f,1.f};
	//+1 for the constraints
	int i;
	for(i = 0; i < problemwidth;i++) {
		for(int j; j < rows;j++) {
			prob.matrix[i][j] = 0.0 + ((float) !!(set(j) & bids[i]));
			printf("%.1f\n",prob.matrix[i][j]);
		}
		prob.cost[i] = -value[i];
	}
	//put the I matrix in mem
	for(; i < problemwidth+rows;i++) {
		for(int j =0; j < rows;j++) {
			prob.matrix[i][j] = 0.0f;
			if(j == i-problemwidth) {
				prob.matrix[i][j] = 1.0f;
			}
		}
	}

	//set the constraints
	for(int j=0; j < rows;j++) {
		prob.matrix[problemwidth+rows][j] = 1.0f;
	}

	//CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(float) * 26843545));
	printf("hello\n");
	//do_simplex<<<14,1024>>>(d,26843545);
	//CUDA_CHECK_RETURN(cudaThreadSynchronize());
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

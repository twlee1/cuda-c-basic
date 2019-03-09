#include <stdio.h>
#include <stdlib.h>

#define N (2048*2048)		// 2048*2048=4194304
#define THREADS_PER_BLOCK 512

__global__ void add(int *d_a, int *d_b, int *d_c){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	d_c[index] = d_a[index] + d_b[index];
}

void random_ints(int* x, int size){
	for (int i=0; i<size; i++){
		x[i]=rand()%10;
	}
}

int main (void) {
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	// Alloc speace for host copies of a, b, c and setup input values
	a = (int*)malloc(size); random_ints(a, N);
	b = (int*)malloc(size); random_ints(b, N);
	c = (int*)malloc(size); 

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add<<< N/THREADS_PER_BLOCK , THREADS_PER_BLOCK >>>(d_a, d_b, d_c);
	// N / THREADS_PER_BLOCK = 4184304 / 512 = 8192 BLOKCS (Why 8192?)
	// THREADS_PER_BLCOK = 512 THREDS per BLOCK (Why 512? Is this optimal value?)
	// How can we choose optimal thread block configuation?
	// Warp = 32 threads. GPU executes threads at a warp level. 512 threds = 16 warps.
        // So, 16 warps are assgined to each SM.
	
	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	// Print reuslts
	//for (int i=0; i<N; i++){
	//	printf("[%d] a:%d + b:%d = c:%d\n", i, a[i], b[i], c[i]);
	//}

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;
}

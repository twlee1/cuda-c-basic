#include <stdio.h>
#include <stdlib.h>

#define RADIUS 3
#define BLK_SIZE 256
#define NUM_ELEMENTS (BLK_SIZE * 32)	// 256 * 32 = 8192


__global__ void stencil_1d(int *d_in, int *d_out){
	int gindex = (blockIdx.x * blockDim.x) + threadIdx.x + RADIUS;

	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; offset++)
		result += d_in[gindex + offset];

	d_out[gindex - RADIUS] = result;
}


int main(void){
	int h_in[ NUM_ELEMENTS + (2*RADIUS) ];
	int h_out[ NUM_ELEMENTS ];	
	int *d_in, *d_out;

	// Initialize host input values
	for (int i=0; i<(NUM_ELEMENTS + 2*RADIUS); i++)
		h_in[i] = 1;

	// Allocate device global memory
	cudaMalloc( &d_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int) );
	cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int) );

	// Copy HOST -> DEVICE
	cudaMemcpy( d_in, h_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int), cudaMemcpyHostToDevice);

	// Launch kernel
	stencil_1d<<< NUM_ELEMENTS/BLK_SIZE, BLK_SIZE>>>(d_in, d_out);
	// NUM_ELEMENTS / BLK_SIZE = 8192 / 256 = 32
	// BLK_SIZE = 256

	// Copy result DEVICE -> HOST
	cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

	// Verify results
	int err_cnt = 0;
	for (int i=0; i<NUM_ELEMENTS; i++){
		if (h_out[i] != 7){
			printf("h_out[%d] == %d != 7\n", i, h_out[i]);
			err_cnt++;
			break;
		}
	}
	if (err_cnt!=0){ 
		printf("Wrong result\n"); 
	}else{
		printf("Success\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}

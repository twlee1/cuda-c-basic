#include <stdio.h>

//__global__ void mykernel(void) {i
__global__ void add(int *a, int *b, int *c){
	*c = *a + *b;	
}

int main (void) {

	int a, b, c;
	int *d_a, *d_b, *d_c;
	int size=sizeof(int);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	a=2;
	b=7;
	
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	add<<<1,1>>>(d_a, d_b, d_c);

	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	printf("c=%d\n", c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


	//mykernel<<<1,1>>>();

	//printf("Hello world\n");
	return 0;
}

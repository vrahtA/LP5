//Cell 1

%%writefile vector.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void addVectors(int *A, int *B, int *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
        C[i] = A[i] + B[i];
}

int main()
{
    int n = 10;
    int size = n * sizeof(int);

    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    A = new int[n];
    B = new int[n];
    C = new int[n];

    for(int i = 0; i < n; i++)
    {
        A[i] = i;
        B[i] = i * 2;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    addVectors<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++)
        cout << C[i] << " ";

    return 0;
}

//Cell 2
!nvcc vector.cu -o vector

//Cell 3
!./vector

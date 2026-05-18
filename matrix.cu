//Cell 1
%%writefile matrix.cu
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void matmul(int *A, int *B, int *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N)
    {
        int sum = 0;

        for(int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main()
{
    int N = 4;
    int size = N * N * sizeof(int);

    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    A = new int[N * N];
    B = new int[N * N];
    C = new int[N * N];

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize matrices
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            A[i * N + j] = 1;
            B[i * N + j] = 1;
        }
    }

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks(1,1);

    matmul<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result Matrix:\n";

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cout << C[i * N + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

//Cell 2
!nvcc matrix.cu -o matrix

//Cell 3
!./matrix

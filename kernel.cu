#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <stdio.h>
#include<iostream>
#include<cstdlib>
#include<chrono>
#include<assert.h>

using namespace std;
//izracun 
__global__ void multiply(int* a, int* b, int* c, int n) {
	//izracun retka i stupca za svaki thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//provjera granica
	if ((row < n) && (col < n)) {
		int tmp = 0;
		for (int i = 0; i < n; i++) {
			tmp += a[row * n + i] * b[i * n + col];
		}

		c[row * n + col] = tmp;
	}
}

//izvodenje na procesoru
void runOnHost(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				c[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}
}

//provjera rezultata
void verify(int* a, int* b, int* c, int n) {

	cout << "Provjera rezultata: \n";

	int tmp;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			tmp = 0;
			for (int k = 0; k < n; k++) {
				tmp += a[i * n + k] * b[k * n + j];
			}
			if (c[i * n + j] != tmp)
				cout << "nije dobro" << endl;
		}
	}
}

void init_matrices(int* m, int n) {
	for (int i = 0; i < n * n; i++) {
		m[i] = rand() % 10;
	}
}

int main() {
	int N = 1 << 10; //dimenzije matrica, 1024x1024, GPU postaje brži na N = 1 << 6
	size_t bytes = N * N * sizeof(int);

	//host pointeri
	int* h_a, * h_b, * h_c, * h_c_cpu;//rezultantna matrica za izracun na hostu

	//device pointeri
	int* d_a, * d_b, * d_c;

	//alokacija host memorije
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);
	h_c_cpu = (int*)malloc(bytes);

	//alokacija device memorije
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	//inicijalizacija matrica
	init_matrices(h_a, N);
	init_matrices(h_b, N);

	//kopiranje matrica na GPU
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	int BLOCK_SIZE = 16;
	int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	//pocetak mjerenja za GPU
	chrono::steady_clock::time_point begin = chrono::steady_clock::now();

	//launch kernel
	multiply << <grid, threads >> > (d_a, d_b, d_c, N);
	cudaDeviceSynchronize();



	//kopiraj rezultat sa GPU-a na host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	//zavrsetak mjerenja za GPU
	chrono::steady_clock::time_point end = chrono::steady_clock::now();

	cout << "Vrijeme za izracun na GPU = " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " mikrosekundi" << endl;

	//pocetak mjerenja za CPU
	chrono::steady_clock::time_point beginCPU = chrono::steady_clock::now();

	//izracun na CPU
	runOnHost(h_a, h_b, h_c_cpu, N);

	//kraj mjerenja za CPU
	chrono::steady_clock::time_point endCPU = chrono::steady_clock::now();

	cout << "Vrijeme za izracun na CPU = " << chrono::duration_cast<std::chrono::milliseconds>(endCPU - beginCPU).count() << " mikrosekundi" << endl;

	verify(h_a, h_b, h_c, N);

	cout <<"\neverything works" << endl;

	//ispis dobivene matrice
	/*for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << h_c[i * N + j] << " ";
		}
		cout << endl;
	}*/

	/*for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << h_c_cpu[i * N + j] << " ";
		}
		cout << endl;
	}*/

	//oslobadanje memorije na hostu
	free(h_a);
	free(h_b);
	free(h_c);

	//oslobadanje memorije na GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}


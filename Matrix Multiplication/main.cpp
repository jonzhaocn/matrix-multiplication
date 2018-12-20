#include<iostream>
#include "matrix.h"
#include<stdio.h>
#include<windows.h>
using namespace std;
int main() 
{
	int n;
	int lower_bound = 0;
	int upper_bound = 1;
	srand((unsigned int)time(NULL));
	cout << "please input a int as the matrix size:";
	cin >> n;
	if (n <= 0) 
	{
		printf("matrix size should greater than 0.\n");
		system("pause");
		return -1;
	}
	cout<<"matrix size:"<<n<<endl;
	float **matrix_a = create_random_matrix(n, lower_bound, upper_bound);
	float **matrix_b = create_random_matrix(n, lower_bound, upper_bound);
	float **result1 = create_zeros_matrix(n);
	float **result2 = create_zeros_matrix(n);
	LARGE_INTEGER freq;
	LARGE_INTEGER time_start;
	LARGE_INTEGER time_end;
	QueryPerformanceFrequency(&freq);
	//ordinary matrix multiplication
	QueryPerformanceCounter(&time_start);
	matrix_multiplication(n, result1, matrix_a, matrix_b);
	QueryPerformanceCounter(&time_end);
	double time = (double)(time_end.QuadPart - time_start.QuadPart) / (double)freq.QuadPart;
	printf("\n----------ordinary matrix multiplication----------\n");
	printf("Time taken %0.6f s\n", time);

	//avx matrix multiplication
	QueryPerformanceCounter(&time_start);
	matrix_multiplication_simd(n, result2, matrix_a, matrix_b);
	QueryPerformanceCounter(&time_end);
	double avx_time = (double)(time_end.QuadPart - time_start.QuadPart) / (double)freq.QuadPart;
	printf("\n----------avx matrix multiplication----------\n");
	printf("Time taken %0.3f s\n", avx_time);

	//speed up rate
	printf("speed up rate: %f\n", time/avx_time);
	float ssd = matrixs_sum_squared_difference(n, result1, result2);
	printf("sum of squared difference: %f\n", ssd);
	
	set_matrix_zero(n, result2);
	//further optimized matrix multiplication
	QueryPerformanceCounter(&time_start);
	matrix_multiplication_multi_thread(n, result2, matrix_a, matrix_b);
	QueryPerformanceCounter(&time_end);
	double opt_time = (double)(time_end.QuadPart - time_start.QuadPart) / (double)freq.QuadPart;
	printf("\n----------further optimized multiplication----------\n");
	printf("Time taken %0.3f s\n", opt_time);

	//speed up rate
	printf("speed up rate: %f\n", time / opt_time);
	ssd = matrixs_sum_squared_difference(n, result1, result2);
	printf("sum of squared difference: %f\n", ssd);
	

	//free matrix memory
	matrix_delete(n, matrix_a);
	matrix_delete(n, matrix_b);
	matrix_delete(n, result1);
	matrix_delete(n, result2);
	system("pause");
	return 0;
}

#include<iostream>
#include<time.h>
#include "matrix.h"
#include<stdio.h>
using namespace std;
int main() 
{
	int n;
	int lower_bound = 0;
	int upper_bound = 100;
	srand((unsigned int)time(NULL));
	cout << "please input a int as the matrix size:";
	cin >> n;
	cout<<"matrix size:"<<n<<endl;
	float **matrix_a = create_random_matrix(n, lower_bound, upper_bound);
	float **matrix_b = create_random_matrix(n, lower_bound, upper_bound);
	float **result1 = create_zeros_matrix(n);
	float **result2 = create_zeros_matrix(n);
	
	//ordinary matrix multiplication
	clock_t start = clock();
	matrix_multiplication(n, result1, matrix_a, matrix_b);
	clock_t end = clock();
	float time = (((float)end - (float)start) / CLOCKS_PER_SEC);
	printf("\n--------------------ordinary matrix multiplication\n");
	printf("Time taken %0.3f s\n", time);

	//avx matrix multiplication
	start = clock();
	simd_matrix_multiplication(n, result2, matrix_a, matrix_b);
	end = clock();
	float avx_time = (((float)end - (float)start) / CLOCKS_PER_SEC);
	printf("\n--------------------avx matrix multiplication\n");
	printf("Time taken %0.3f s\n", avx_time);

	//speed up rate
	printf("speed up rate: %f\n", time/avx_time);
	float ssd = matrixs_sum_squared_difference(n, result1, result2);
	printf("sum of squared difference: %f\n", ssd);

	//optimized matrix multiplication
	start = clock();
	further_optimized_matrix_multiplication(n, result2, matrix_a, matrix_b);
	end = clock();
	float opt_time = (((float)end - (float)start) / CLOCKS_PER_SEC);
	printf("\n--------------------futher optimized multiplication\n");
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

#include "utils.h"
float **create_random_matrix(int size, int lower_bound, int upper_bound)
{
	float **matrix;
	matrix = new float*[size];
	for (int i = 0; i < size; i++)
	{
		// a avx register can store 8 float, make the matrix[i] be 32 byte alignment
		matrix[i] = (float *)_aligned_malloc(sizeof(float)*size, sizeof(float) * 8);
		for (int j = 0; j < size; j++)
		{
			matrix[i][j] = float(rand()/double(RAND_MAX)) * upper_bound + lower_bound;
		}
	}
	return matrix;
}
float **create_zeros_matrix(int size)
{
	float **matrix;
	matrix = new float*[size];
	for (int i = 0; i < size; i++)
	{
		// a avx register can store 8 float, make the matrix[i] be 32 byte alignment
		matrix[i] = (float *)_aligned_malloc(sizeof(float)*size, sizeof(float) * 8);
		memset(matrix[i], 0, sizeof(float)*size);
	}
	return matrix;
}
void set_matrix_zero(int size, float **matrix)
{
	for (int i = 0; i < size; i++)
	{
		memset(matrix[i], 0, sizeof(float)*size);
	}
}
void matrix_transpose(int size, float **matrix)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			std::swap(matrix[i][j], matrix[j][i]);
		}
	}
}
float matrixs_sum_squared_difference(int size, float **mat_a, float **mat_b)
{
	float result = 0;
	float temp;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			temp = mat_a[i][j] - mat_b[i][j];
			result += temp*temp;
		}
	}
	return result;
}
void matrix_print(int size, float **mat)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			printf("%1.2f ", mat[i][j]);
		}
		printf("\n");
	}
}
void matrix_delete(int size, float **mat) 
{
	for (int i = 0; i < size; i++) 
	{
		_aligned_free(mat[i]);
	}
	delete [] mat;
}
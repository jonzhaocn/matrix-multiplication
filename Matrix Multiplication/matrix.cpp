#include "matrix.h"
#include <intrin.h>
#include "threadpool.h"
std::mutex mutex;
//baseline
void matrix_multiplication(int n, float **mat_c, float **mat_a, float **mat_b)
{
	for (int i = 0; i < n; i++)
	{
		//loop permutation
		//set k as the second loop index will speed up the matrix multiplication
		for (int k = 0; k < n; k++)
		{
			for (int j = 0; j < n; j++)
			{
				mat_c[i][j] += mat_a[i][k] * mat_b[k][j];
			}
		}
	}
}
//avx matrix multiplication
void matrix_multiplication_simd(int size, float **mat_c, float **mat_a, float **mat_b)
{
	__m256 m256_a, m256_b, m256_c;
	__m256i mask = _mm256_setzero_si256();
	int block_width = 8;
	int len = size / block_width;
	//if remaider is not equal to zero, set a mask for _mm256_maskload_ps
	int remainder = size % block_width;
	if (remainder > 0) 
	{
		int*temp = (int*)&mask;
		for (int i = 0; i < remainder; i++) 
		{
			temp[i] = -1;
		}
	}
	// matrix transpose
	matrix_transpose(size, mat_b);
	for (int i = 0; i < size; i += 1)
	{
		for (int j = 0; j < size; j += 1)
		{
			int k = 0;
			m256_c = _mm256_setzero_ps();
			for (; k < block_width*len; k += block_width)
			{
				m256_a = _mm256_load_ps(mat_a[i]+k);
				m256_b = _mm256_load_ps(mat_b[j]+k);
				//m256_c = _mm256_fmadd_ps(m256_a, m256_b, m256_c);
				m256_c = _mm256_add_ps(m256_c, _mm256_mul_ps(m256_a, m256_b));
			}
			if (remainder > 0) 
			{
				m256_a = _mm256_maskload_ps(mat_a[i]+k, mask);
				m256_b = _mm256_maskload_ps(mat_b[j]+k, mask);
				//m256_c = _mm256_fmadd_ps(m256_a, m256_b, m256_c);
				m256_c = _mm256_add_ps(m256_c, _mm256_mul_ps(m256_a, m256_b));
			}
			const float *temp = (const float*)&m256_c;
			mat_c[i][j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
		}
	}
	matrix_transpose(size, mat_b);
}
void matrix_multiplication_single_thread(int size, float **mat_c, float **mat_a, float **mat_b, int row_start, int row_end) 
{
	__m256 m256_multi, m256_a1, m256_a2, m256_b1, m256_b2, m256_c1, m256_c2, m256_d1, m256_d2, m256_e1, m256_e2, 
		m256_f1, m256_f2, m256_g1, m256_g2, m256_h1, m256_h2;
	__m256i mask = _mm256_setzero_si256();
	int block_width = 8;
	int len = size / block_width;
	//if remaider is not equal to zero, set a mask for _mm256_maskload_ps
	int remainder = size % block_width;
	if (remainder > 0)
	{
		int*temp = (int*)&mask;
		for (int i = 0; i < remainder; i++)
		{
			temp[i] = -1;
		}
	}
	for (int i = row_start; i < row_end; i += 1)
	{
		int j = 0;
		int unwinding_count = 8;
		int loop_times = size / unwinding_count;
		for (; j < loop_times*unwinding_count; j += unwinding_count)
		{
			int k = 0;
			m256_a2 = _mm256_setzero_ps();
			m256_b2 = _mm256_setzero_ps();
			m256_c2 = _mm256_setzero_ps();
			m256_d2 = _mm256_setzero_ps();
			m256_e2 = _mm256_setzero_ps();
			m256_f2 = _mm256_setzero_ps();
			m256_g2 = _mm256_setzero_ps();
			m256_h2 = _mm256_setzero_ps();
			for (; k < block_width*len; k += block_width)
			{
				//using the same multiplier from mat_c, reduce the times of loading memory
				//loop unwinding
				m256_multi = _mm256_load_ps(mat_a[i] + k);

				m256_a1 = _mm256_load_ps(mat_b[j] + k);
				m256_a2 = _mm256_add_ps(m256_a2, _mm256_mul_ps(m256_multi, m256_a1));
				m256_b1 = _mm256_load_ps(mat_b[j + 1] + k);
				m256_b2 = _mm256_add_ps(m256_b2, _mm256_mul_ps(m256_multi, m256_b1));
				m256_c1 = _mm256_load_ps(mat_b[j + 2] + k);
				m256_c2 = _mm256_add_ps(m256_c2, _mm256_mul_ps(m256_multi, m256_c1));
				m256_d1 = _mm256_load_ps(mat_b[j + 3] + k);
				m256_d2 = _mm256_add_ps(m256_d2, _mm256_mul_ps(m256_multi, m256_d1));
				m256_e1 = _mm256_load_ps(mat_b[j + 4] + k);
				m256_e2 = _mm256_add_ps(m256_e2, _mm256_mul_ps(m256_multi, m256_e1));
				m256_f1 = _mm256_load_ps(mat_b[j + 5] + k);
				m256_f2 = _mm256_add_ps(m256_f2, _mm256_mul_ps(m256_multi, m256_f1));
				m256_g1 = _mm256_load_ps(mat_b[j + 6] + k);
				m256_g2 = _mm256_add_ps(m256_g2, _mm256_mul_ps(m256_multi, m256_g1));
				m256_h1 = _mm256_load_ps(mat_b[j + 7] + k);
				m256_h2 = _mm256_add_ps(m256_h2, _mm256_mul_ps(m256_multi, m256_h1));
			}
			if (remainder > 0)
			{
				m256_multi = _mm256_maskload_ps(mat_a[i] + k, mask);

				m256_a1 = _mm256_maskload_ps(mat_b[j] + k, mask);
				m256_a2 = _mm256_add_ps(m256_a2, _mm256_mul_ps(m256_multi, m256_a1));
				m256_b1 = _mm256_maskload_ps(mat_b[j + 1] + k, mask);
				m256_b2 = _mm256_add_ps(m256_b2, _mm256_mul_ps(m256_multi, m256_b1));
				m256_c1 = _mm256_maskload_ps(mat_b[j + 2] + k, mask);
				m256_c2 = _mm256_add_ps(m256_c2, _mm256_mul_ps(m256_multi, m256_c1));
				m256_d1 = _mm256_maskload_ps(mat_b[j + 3] + k, mask);
				m256_d2 = _mm256_add_ps(m256_d2, _mm256_mul_ps(m256_multi, m256_d1));
				m256_e1 = _mm256_maskload_ps(mat_b[j + 4] + k, mask);
				m256_e2 = _mm256_add_ps(m256_e2, _mm256_mul_ps(m256_multi, m256_e1));
				m256_f1 = _mm256_maskload_ps(mat_b[j + 5] + k, mask);
				m256_f2 = _mm256_add_ps(m256_f2, _mm256_mul_ps(m256_multi, m256_f1));
				m256_g1 = _mm256_maskload_ps(mat_b[j + 6] + k, mask);
				m256_g2 = _mm256_add_ps(m256_g2, _mm256_mul_ps(m256_multi, m256_g1));
				m256_h1 = _mm256_maskload_ps(mat_b[j + 7] + k, mask);
				m256_h2 = _mm256_add_ps(m256_h2, _mm256_mul_ps(m256_multi, m256_h1));

			}
			const float *temp1 = (const float*)&m256_a2;
			float temp1_result = temp1[0] + temp1[1] + temp1[2] + temp1[3] + temp1[4] + temp1[5] + temp1[6] + temp1[7];
			const float *temp2 = (const float*)&m256_b2;
			float temp2_result = temp2[0] + temp2[1] + temp2[2] + temp2[3] + temp2[4] + temp2[5] + temp2[6] + temp2[7];
			const float *temp3 = (const float*)&m256_c2;
			float temp3_result = temp3[0] + temp3[1] + temp3[2] + temp3[3] + temp3[4] + temp3[5] + temp3[6] + temp3[7];
			const float *temp4 = (const float*)&m256_d2;
			float temp4_result = temp4[0] + temp4[1] + temp4[2] + temp4[3] + temp4[4] + temp4[5] + temp4[6] + temp4[7];
			const float *temp5 = (const float*)&m256_e2;
			float temp5_result = temp5[0] + temp5[1] + temp5[2] + temp5[3] + temp5[4] + temp5[5] + temp5[6] + temp5[7];
			const float *temp6 = (const float*)&m256_f2;
			float temp6_result = temp6[0] + temp6[1] + temp6[2] + temp6[3] + temp6[4] + temp6[5] + temp6[6] + temp6[7];
			const float *temp7 = (const float*)&m256_g2;
			float temp7_result = temp7[0] + temp7[1] + temp7[2] + temp7[3] + temp7[4] + temp7[5] + temp7[6] + temp7[7];
			const float *temp8 = (const float*)&m256_h2;
			float temp8_result = temp8[0] + temp8[1] + temp8[2] + temp8[3] + temp8[4] + temp8[5] + temp8[6] + temp8[7];
			{
				std::lock_guard<std::mutex> lck(mutex);
				mat_c[i][j] = temp1_result;
				mat_c[i][j + 1] = temp2_result;
				mat_c[i][j + 2] = temp3_result;
				mat_c[i][j + 3] = temp4_result;
				mat_c[i][j + 4] = temp5_result;
				mat_c[i][j + 5] = temp6_result;
				mat_c[i][j + 6] = temp7_result;
				mat_c[i][j + 7] = temp8_result;
			}
		}
		for (; j < size; j++)
		{
			int k = 0;
			m256_a2 = _mm256_setzero_ps();
			for (; k < block_width*len; k += block_width)
			{
				m256_multi = _mm256_load_ps(mat_a[i] + k);
				m256_a1 = _mm256_load_ps(mat_b[j] + k);
				m256_a2 = _mm256_add_ps(m256_a2, _mm256_mul_ps(m256_multi, m256_a1));
			}
			if (remainder > 0)
			{
				m256_multi = _mm256_maskload_ps(mat_a[i] + k, mask);
				m256_a1 = _mm256_maskload_ps(mat_b[j] + k, mask);
				m256_a2 = _mm256_add_ps(m256_a2, _mm256_mul_ps(m256_multi, m256_a1));
			}
			const float *temp = (const float*)&m256_a2;
			float temp_result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
			{
				std::lock_guard<std::mutex> lck(mutex);
				mat_c[i][j] = temp_result;
			}
		}
	}
}
void matrix_multiplication_multi_thread(int size, float **mat_c, float **mat_a, float **mat_b)
{
	unsigned short thread_count = 8;
	// threadpool
	std::threadpool executor{ thread_count};
	std::vector< std::future<void> > results;
	int i = 0;
	int step = int(size / thread_count);
	// transpose matrix before multiplication and the mat_a and mat_b can be assess sequentially
	matrix_transpose(size, mat_b);
	for (; i < step*thread_count; i+=step)
	{
		results.emplace_back(executor.commit(matrix_multiplication_single_thread, size, mat_c, mat_a, mat_b, i, i + step));
	}
	if (i < size)
	{
		results.emplace_back(executor.commit(matrix_multiplication_single_thread, size, mat_c, mat_a, mat_b, i, size));
	}
	for (auto && result : results)
	{
		result.get();
	}
	matrix_transpose(size, mat_b);
}
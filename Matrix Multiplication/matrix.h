#pragma once
#include "utils.h"
#include <thread>
#include <mutex>
#include <vector>
void matrix_multiplication(int n, float **mat_c, float **mat_a, float **mat_b);
void matrix_multiplication_simd(int size, float **mat_c, float **mat_a, float **mat_b);
void matrix_multiplication_single_thread(int size, float **mat_c, float **mat_a, float **mat_b, int row_start, int row_end);
void matrix_multiplication_multi_thread(int size, float **mat_c, float **mat_a, float **mat_b);
#pragma once
#include "utils.h"
void matrix_multiplication(int n, float **mat_c, float **mat_a, float **mat_b);
void simd_matrix_multiplication(int size, float **mat_c, float **mat_a, float **mat_b);
void optimized_matrix_multiplication(int size, float **mat_c, float **mat_a, float **mat_b);

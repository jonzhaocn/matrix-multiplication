#pragma once
#include<string.h>
#include<stdio.h>
#include<iostream>
#include <time.h>
float ** create_random_matrix(int size, int lower_bound, int upper_bound);
float ** create_zeros_matrix(int size);
void matrix_transpose(int n, float **mat);
float matrixs_sum_squared_difference(int size, float **mat_a, float **mat_b);
void matrix_print(int size, float **mat);
void matrix_delete(int size, float **mat);
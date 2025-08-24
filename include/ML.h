#ifndef ML_H
#define ML_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <matrix/matrix.h>

typedef struct
{
    size_t count;
    c_matrix *ws, *bs, *as;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float sigmoidf(float x);
void mat_sigmoid(c_matrix *m);

// size_t arch[] = {2, 2, 1};
// NN nn = nn_alloc(arch, ARRAY_LEN(arch));

NN alloc_nn(size_t *arch, size_t arch_count);
void zero_nn(NN *nn);
void rand_nn(NN *nn, float low, float high);
void print_nn(NN *nn, const char *name);
#define PRINT_NN(nn) print_nn(&nn, #nn);

void forward_nn(NN *nn);
float cost_nn(NN *nn, c_matrix *in, c_matrix *out);
void diff_nn(NN *nn, NN *g, float eps, c_matrix *in, c_matrix *out);
void backprop_nn(NN *nn, NN *g, c_matrix *in, c_matrix *out);
void learn_nn(NN *nn, NN *g, float rate);

#endif
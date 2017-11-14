#include "shuffle_layer.h"

//#include "utils.h"
#include "cuda.h"
#include "blas.h"
//#include "gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//***********TODO: 参考原论文，shuffle operation相当于矩阵转置，可直接用BLAS********twq
shuffle_layer make_shuffle_layer(int batch, int h, int w, int c, int group)
{
    int i;
	shuffle_layer l = {0};
    l.type = SHUFFLE;

	l.h = h;
	l.w = w;
	l.c = c;
    l.inputs = w * h * c;
    l.outputs = l.inputs;
    l.batch=batch;
	l.group_ = group;

    l.out_h = h;
    l.out_w = w;
    l.out_c = c;

    l.output = calloc(batch*l.outputs, sizeof(float));
	l.delta = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_shuffle_layer;
    l.backward = backward_shuffle_layer;

#ifdef GPU
    l.forward_gpu = forward_shuffle_layer_gpu;
    l.backward_gpu = backward_shuffle_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
	l.delta_gpu = cuda_make_array(l.delta, l.outputs*batch);
#endif
    fprintf(stderr, "shuffle                 %4d x%4d x%4d   ->  %4d x%4d x%4d; group : %d\n", w,h,c,w,h,c, group);
    return l;
}

void resize_shuffle_layer(shuffle_layer *l, int w, int h)
{
	l->h = h;
	l->w = w;

	l->inputs = w * h * l->c;
	l->outputs = l->inputs;

	l->out_h = h;
	l->out_w = w;

	l->output = realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = realloc(l->delta, l->batch*l->outputs * sizeof(float));

#ifdef GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
	l->delta_gpu = cuda_make_array(l->delta, l->outputs*l->batch);
#endif
}

void shuffle_cpu(float *output, float *input, int group_row, int group_column, int map_size)
{
	for (int i = 0; i < group_row; ++i) // 2
	{
		for (int j = 0; j < group_column; ++j) // 3
		{
			float* p_i = input + (i * group_column + j) * map_size;
			float* p_o = output + (j * group_row + i) * map_size;

			copy_cpu(map_size, p_i, 1, p_o, 1);
		}
	}
}

void shuffle_gpu(float *output, float *input, int group_row, int group_column, int map_size)
{
	for (int i = 0; i < group_row; ++i) // 2
	{
		for (int j = 0; j < group_column; ++j) // 3
		{
			float* p_i = input + (i * group_column + j) * map_size;
			float* p_o = output + (j * group_row + i) * map_size;

			copy_gpu(map_size, p_i, 1, p_o, 1);
		}
	}
}

void forward_shuffle_layer(shuffle_layer l, network state)
{
    int i;
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    float *input = state.input;
    float *output = l.output;

	const int batch = l.batch;
	const int out_size = l.outputs;
	const int map_size = l.h * l.w;
	const int chs = l.c;

	int group_row = l.group_;
	int group_col = chs / group_row;
	//CHECK_EQ(chs, (group_column * group_row)) << "Wrong group size.";

    for(i = 0; i < l.batch; ++i){
		shuffle_cpu(output + i*out_size, input + i*out_size, group_row, group_col, map_size);
    }
	for (i = 0; i < l.outputs; i++) printf("%f ", output[i]);
}

void backward_shuffle_layer(shuffle_layer l, network state)
{
	int i;
	fill_cpu(l.outputs*l.batch, 0, l.output, 1);
	float *input = l.delta;
	float *delta = state.delta;

	const int batch = l.batch;
	const int out_size = l.inputs;
	const int map_size = l.h * l.w;
	const int chs = l.c;

	int group_col = l.group_;
	int group_row = chs / group_col;
	//CHECK_EQ(chs, (group_column * group_row)) << "Wrong group size.";

	for (i = 0; i < l.batch; ++i) {
		shuffle_cpu(delta + i*out_size, input + i*out_size, group_row, group_col, map_size);
	}
	for (i = 0; i < l.inputs; i++) printf("%f ", delta[i]);
}

#ifdef GPU

void forward_shuffle_layer_gpu(shuffle_layer l, network state)
{
	int i;
	fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
	float *input = state.input_gpu;
	float *output = l.output_gpu;

	const int batch = l.batch;
	const int out_size = l.outputs;
	const int map_size = l.h * l.w;
	const int chs = l.c;

	int group_row = l.group_;
	int group_col = chs / group_row;
	//CHECK_EQ(chs, (group_column * group_row)) << "Wrong group size.";
	for (i = 0; i < l.batch; ++i) {
		shuffle_gpu(output + i*out_size, input + i*out_size, group_row, group_col, map_size);
	}
}

void backward_shuffle_layer_gpu(shuffle_layer l, network state)
{
	int i;
	fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
	float *input = l.delta_gpu;
	float *delta = state.delta_gpu;

	const int batch = l.batch;
	const int out_size = l.outputs;
	const int map_size = l.h * l.w;
	const int chs = l.c;

	int group_col = l.group_;
	int group_row = chs / group_col;
	//CHECK_EQ(chs, (group_column * group_row)) << "Wrong group size.";
	if (delta) {
		for (i = 0; i < l.batch; ++i) {
			shuffle_gpu(delta + i*out_size, input + i*out_size, group_row, group_col, map_size);
		}
	}
}
#endif

void test_shuffle_forward()
{
	shuffle_layer l = make_shuffle_layer(1, 2, 2, 12, 4);
	float data[] = { 
		0,0,0,0,
		1,1,1,1,
		2,2,2,2,
		3,3,3,3,
		4,4,4,4,
		5,5,5,5,
		6,6,6,6,
		7,7,7,7,
		8,8,8,8,
		9,9,9,9,
		10,10,10,10,
		11,11,11,11
	};
	network state = { 0 };
	state.input = data;
	forward_shuffle_layer(l, state);
}

void test_shuffle_backward()
{
	shuffle_layer l = make_shuffle_layer(1, 2, 2, 12, 4);
	float data[] = {
		0,0,0,0,
		3,3,3,3,
		6,6,6,6,
		9,9,9,9,
		1,1,1,1,
		4,4,4,4,
		7,7,7,7,
		10,10,10,10,
		2,2,2,2,
		5,5,5,5,
		8,8,8,8,
		11,11,11,11
	};
	network state = { 0 };
	state.delta = calloc(1, 1000);
	l.delta = data;
	backward_shuffle_layer(l, state);
}

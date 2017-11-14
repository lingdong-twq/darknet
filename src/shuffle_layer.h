/*****************	Implement of Shuffle Layer*********************** 
// Reference: ShuffleNet ( https://arxiv.org/pdf/1707.01083.pdf )
// Created on 2017 / 10 / 31  by twq
/*********************************************************************/
#ifndef SHUFFLE_LAYER_H
#define SHUFFLE_LAYER_H

#include "layer.h"
#include "network.h"
typedef layer shuffle_layer;

shuffle_layer make_shuffle_layer(int batch, int h, int w, int c, int group);
void forward_shuffle_layer(shuffle_layer layer, network state);
void backward_shuffle_layer(shuffle_layer layer, network state);

void resize_shuffle_layer(shuffle_layer *layer, int w, int h);
void shuffle_cpu(float *output, float *input, int group_row, int group_column, int map_size);

#ifdef GPU
void forward_shuffle_layer_gpu(shuffle_layer layer, network state);
void backward_shuffle_layer_gpu(shuffle_layer layer, network state);
void shuffle_gpu(float *output, float *input, int group_row, int group_column, int map_size);
#endif

void test_shuffle_forward();
void test_shuffle_backward();
#endif

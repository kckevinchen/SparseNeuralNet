import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
from tqdm import tqdm
from utils import *
import tensorflow as tf
from train import load_cifar100
"""
0 -- dense: keras conv2D
1 -- sparse + dense: masked keras conv2D
2 -- partial sparse: sparse conv1x1 + dense masked conv3x3
3 -- fully sparse: sparse conv1x1 + dense 1x1
"""
@tf.function
def forward_pass(model, inputs):
	model(inputs)

def load_model_layers(sparse_level):
	model = resnet50(num_classes=100, sparse_level=sparse_level, save_input_shape=True)

	#load mtx
	model.build((16,3,32,32))
	forward_pass(model, np.random.randn(16,3,32,32))

	if sparse_level != 0:
		restore_from_mtx(model, "./mtx_229_0.1")
	edge_layers = flatten_layers(model)
	return np.array(list(filter(lambda layer: isinstance(layer, GeneralConv2D), edge_layers)))
	
def get_layers_conv1x1():
	"""
	Return three layers, 
	conv1x1 using the tf.keras.layers.Conv2D (sparse level 0)
	conv1x1 using MaksedConv2D (sparse level 1)
	conv1x1 using SparseConv1x1 (sparse level 3 -- fully sparse)
	"""
	layers_level_0 = load_model_layers(0)
	layers_level_1 = load_model_layers(1)
	layers_level_3 = load_model_layers(3)
	indexes = []
	result = {}
	for i, v in enumerate(layers_level_3):
		if "level_3_SparseConv1x1_ksize_1" == v.type:
			indexes.append(i)
	for i in indexes:
		result[i] = (layers_level_0[i], layers_level_1[i], layers_level_3[i])
	return result, indexes
	



def get_layers_conv3x3():
	"""
	Return three layers,
	conv3x3 using the tf.keras.layers.Conv2D (sparse level 0)
	conv3x3 using the MaskedConv2D (sparse level 1)
	conv3x3 using the SparseConv2D (sparse level 3)
	"""
	layers_level_0 = load_model_layers(0)
	layers_level_1 = load_model_layers(1)
	layers_level_3 = load_model_layers(3)
	indexes = []
	result = {}
	for i, v in enumerate(layers_level_3):
		if "level_3_SparseConv2D_ksize_3" == v.type:
			indexes.append(i)
	for i in indexes:
		result[i] = (layers_level_0[i], layers_level_1[i], layers_level_3[i])
	return result, indexes

def profile_particular_layer(layer, inputs, num_iter, logdir):
	graph = tf.function(layer)
	with tf.profiler.experimental.Profile(logdir):
		for _ in range(num_iter):
			graph(inputs)


def profile_conv1x1(logdir):
	layers, indexes = get_layers_conv1x1()
	dense, masked_dense, sparse_1x1 = layers[indexes[0]]
	input_val = get_layer_input(dense)
	profile_particular_layer(dense, input_val, 5, "./{}/conv1x1/dense".format(logdir))
	profile_particular_layer(masked_dense, input_val, 5, "./{}/conv1x1/masked_dense".format(logdir))
	profile_particular_layer(sparse_1x1, input_val, 5, "./{}/conv1x1/sparse_1x1".format(logdir))

	
def get_layer_input(layer):
	return np.random.randn(*layer.layer_input_shape)

if __name__ == "__main__":
	profile_conv1x1("profiler_logs")
	# load_model_layers(sparse_level=1)

	# a = resnet50(num_classes=100, sparse_level=0, save_input_shape=True)
	# train_data, val, test = load_cifar100(32, 0.1, True, 123)
	# print(iter(train_data).next())
	# for x, y in train_data:
	# 	print(x.shape)
	# 	break
	# print("cao nai ma")
	# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.ci 
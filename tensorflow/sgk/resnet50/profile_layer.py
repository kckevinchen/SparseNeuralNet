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

	if sparse_level != 0:
		restore_from_mtx(model, "./mtx_229_0.1")
	model.build((16,3,32,32))
	forward_pass(model, np.random.randn(16,3,32,32))
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

def profile_particular_layer_inference(layer, exaust_input, model_input, num_exaust_iter, logdir):
	graph = tf.function(layer)
	for _ in range(num_exaust_iter):
		graph(exaust_input)
	with tf.profiler.experimental.Profile(logdir):
		for x in model_input:
			graph(x)


"""
Just a comment about descriptor bug:
https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
Great explanation!!
"""
def get_steps_function():
	@tf.function
	def steps(batch_data, model, opt):
		"""
		Helper function for profiling the layer
		when training. Since the backward is already defined,
		I'll just use the reduce mean operation for simplicity
		"""
		with tf.GradientTape() as tape:
			loss_value = tf.reduce_mean(model(batch_data))
		gradients = tape.gradient(loss_value, model.trainable_variables)
		opt.apply_gradients(zip(gradients, model.trainable_variables))
	return steps

def profile_particular_layer_train(layer, exaust_input, model_input, num_exaust_iter, logdir):
	opt = tf.keras.optimizers.SGD(learning_rate=0.01)
	steps = get_steps_function()
	for _ in range(num_exaust_iter):
		steps(exaust_input, layer, opt)	

	with tf.profiler.experimental.Profile(logdir):
		for x in tqdm(model_input):
			steps(x, layer, opt)


def profile_conv1x1_inference(logdir):
	layers, indexes = get_layers_conv1x1()
	dense, masked_dense, sparse_1x1 = layers[indexes[0]]
	exaust_input, model_input = get_layer_input(dense, 20)
	profile_particular_layer_inference(dense, exaust_input, model_input, 20, "./{}/conv1x1/dense".format(logdir))
	profile_particular_layer_inference(masked_dense, exaust_input, model_input, 20, "./{}/conv1x1/masked_dense".format(logdir))
	profile_particular_layer_inference(sparse_1x1, exaust_input, model_input, 20, "./{}/conv1x1/sparse_1x1".format(logdir))

def profile_conv1x1_train(logdir):
	layers, indexes = get_layers_conv1x1()
	dense, masked_dense, sparse_1x1 = layers[indexes[0]]
	exaust_input, model_input = get_layer_input(dense, 20)
	profile_particular_layer_train(dense, exaust_input, model_input, 20, "./{}/conv1x1/dense".format(logdir))
	profile_particular_layer_train(masked_dense, exaust_input, model_input, 20, "./{}/conv1x1/masked_dense".format(logdir))
	profile_particular_layer_train(sparse_1x1, exaust_input, model_input, 20, "./{}/conv1x1/sparse_1x1".format(logdir))


def get_layer_input(layer, num_iter):
	input_shape = layer.layer_input_shape
	exaust_input = np.random.randn(*input_shape)
	model_input = np.random.randn(input_shape[0] * num_iter, *input_shape[1:])
	model_input = tf.data.Dataset.from_tensor_slices(model_input)
	model_input = model_input.batch(input_shape[0])
	return exaust_input, model_input

if __name__ == "__main__":
	profile_conv1x1_train("profiler_logs")
	# load_model_layers(sparse_level=1)

	# a = resnet50(num_classes=100, sparse_level=0, save_input_shape=True)
	# train_data, val, test = load_cifar100(32, 0.1, True, 123)
	# print(iter(train_data).next())
	# for x, y in train_data:
	# 	print(x.shape)
	# 	break
	# print("cao nai ma")
	# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.ci 
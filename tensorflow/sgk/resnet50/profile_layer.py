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
		if "level_3_SparseConv1x1_ksize_1" == v.type or "level_3_SparseConv2D_ksize_1" == v.type:
			indexes.append(i)
	for i in indexes:
		result[i] = (layers_level_0[i], layers_level_1[i], layers_level_3[i])
	return result, indexes
	



def get_layers_conv3x3():
	"""
	conv3x3 also includes 3 layers whose kernel = 1 but stride != 1

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
		if "level_3_SparseConv2D_ksize_3" == v.type :
		# if "level_3_sparseconv2d_ksize_3" == v.type:
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
		for x in model_input:
			steps(x, layer, opt)


def profile_conv(logdir, method_type, layer_type, num_layers=1):
	"""
	profile all the conv layers in resnet50 and save in the logdir

	method is either "train" or "inference"
	layer_type is either "conv1x1" or "conv3x3"
	num_layers is the number of layers you want to profile
	"""
	profile_methods = {"train": profile_particular_layer_train, "inference": profile_particular_layer_inference}
	profile_method = profile_methods[method_type]
	layer_methods = {"conv1x1": get_layers_conv1x1, "conv3x3": get_layers_conv3x3}
	layer_method = layer_methods[layer_type]
	layers, indexes = layer_method()
	num_layers = num_layers if num_layers > 0 else len(indexes)
	for i in tqdm(range(num_layers)):
		dense, masked_dense, sparse_1x1 = layers[indexes[i]]
		original_index = indexes[i]
		exaust_input, model_input = get_layer_input(dense, 20)
		#example of the directory:
			#logdir/train/conv1x1/layer_1/dense
		profile_method(dense, exaust_input, model_input, 20, "./{}/{}/{}/layer_{}/dense".format(logdir, method_type, layer_type,original_index))
		profile_method(masked_dense, exaust_input, model_input, 20, "./{}/{}/{}/layer_{}/masked_dense".format(logdir, method_type,layer_type, original_index))
		profile_method(sparse_1x1, exaust_input, model_input, 20, "./{}/{}/{}/layer_{}/sparse".format(logdir, method_type,layer_type, original_index))

def get_layer_input(layer, num_iter):
	input_shape = layer.layer_input_shape
	exaust_input = np.random.randn(*input_shape)
	model_input = np.random.randn(input_shape[0] * num_iter, *input_shape[1:])
	model_input = tf.data.Dataset.from_tensor_slices(model_input)
	model_input = model_input.batch(input_shape[0])
	return exaust_input, model_input

if __name__ == "__main__":
	#example call to profile one conv3x3 layer
	# profile_conv("profiler_logs", "inference", "conv3x3")

	parser = argparse.ArgumentParser(
		description="Training Sparse Resnet.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--logdir", type=str, default = "new_logs")
	parser.add_argument("--method_type", type=str, default="inference")
	parser.add_argument("--layer_type", type=str, default="conv3x3")
	parser.add_argument("--profile_all", type=bool, default=False)
	args = parser.parse_args()
	if args.profile_all:
		ls = ["conv1x1", "conv3x3"]
		ms = ["train", "inference"]
		for l in ls:
			for m in ms:
				print("profiling {} for {}".format(l, m))
				profile_conv(args.logdir, m, l, -1)
	else:
		profile_conv(args.logdir, args.method_type, args.layer_type)

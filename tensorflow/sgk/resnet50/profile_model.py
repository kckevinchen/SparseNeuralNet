from resnet import *
from train import *
import tensorflow as tf
from utils import *
from tqdm import tqdm
import time
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
The code profiles the training of the resnet50 model with differnet sparsity
level over 100 batches, each with batchsize of 32

0 -- dense: keras conv2D
1 -- sparse + dense: masked keras conv2D
2 -- partial sparse: sparse conv1x1 + dense masked conv3x3
3 -- fully sparse: sparse conv1x1 + dense 1x1

I added a new mixed model for profiling. The code needs a bit refactor and I'll do
that later
"""


def get_train_for_epoch():
	"""
	TF graph mode bug, will cause descriptor value error
	This is a work around
	"""
	@tf.function
	def train_for_epoch(model, inputs, targets, opt, loss):
		with tf.GradientTape() as tape:
			y_predict = model(inputs)
			loss_val = loss(y_true=targets, y_pred=y_predict)
		opt.apply_gradients(zip(tape.gradient(
			loss_val, model.trainable_variables), model.trainable_variables))
		return y_predict, loss_val
	return train_for_epoch


@tf.function
def loss(model, inputs, targets, loss_obj):
	y_predict = model(inputs)
	loss_val = loss_obj(y_true=targets, y_pred=y_predict)
	return y_predict, loss_val


def profile_training(sparse_level, logdir, num_iteration=10, mix_layers=False, mix_indexes=[]):
	"""
	The function profiles the training process of resnet50 given the sparse_level
	batchsize is 32 for this profiling

	if use mix model, just set mix_layers to True and provide the mix_indexes (conv indexes where dense is better)
	"""
	DEBUG=False
	batch_size = 32
	mtx_path = "./mtx_229_0.1"
	train_for_epoch = get_train_for_epoch()
	if not mix_layers:
		model = resnet50(num_classes=100, sparse_level=sparse_level)
		if(sparse_level != 0):
			print("Load weight from mtx")
			restore_from_mtx(model, mtx_path)
		model.build((batch_size, 3, 32, 32))
		model(np.random.randn(batch_size, 3, 32, 32))
	else:
		model = resnet50(num_classes=100, sparse_level=3)
		restore_from_mtx(model, mtx_path)
		mix_dense_sparse(model, mix_indexes)
		model.build((batch_size, 3, 32,32))
	if DEBUG:
		layers_expanded = flatten_layers(model) 
		layers_expanded = list(filter(lambda layer: isinstance(layer, GeneralConv2D), layers_expanded))
		count = 0
		for l in layers_expanded:
			if isinstance(l.conv_layer, tf.keras.layers.Conv2D):
				count += 1
		print(count)	
		exit()
	# load data
	train_data, _, _ = load_cifar100(batch_size, 0.1)
	# loss
	loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	# optimizer
	opt = tf.keras.optimizers.SGD(
		learning_rate=0.01, momentum=0.9, nesterov=False, name='SGD')
	run_name = ("resnet50" + '_' + "cifar100" + "_1")

	train_data = train_data.take(min(len(train_data), num_iteration))

	with tf.profiler.experimental.Profile(logdir):
		for x, y in tqdm(train_data):
			train_for_epoch(model, x, y, opt, loss_obj)

def profile_inference(sparse_level, logdir, num_iteration=10, mix_layers=False, mix_indexes=[]):
	batch_size = 32
	mtx_path = "./mtx_229_0.1"
	train_for_epoch = get_train_for_epoch()
	if not mix_layers:
		model = resnet50(num_classes=100, sparse_level=sparse_level)
		if(sparse_level != 0):
			print("Load weight from mtx")
			restore_from_mtx(model, mtx_path)
		model.build((batch_size, 3, 32, 32))
	else:
		model = resnet50(num_classes=100, sparse_level=3)
		restore_from_mtx(model, mtx_path)
		mix_dense_sparse(model, mix_indexes)
		model.build((batch_size, 3, 32,32))

	# load data
	train_data, _, _ = load_cifar100(batch_size, 0.1)
	# loss
	loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	# optimizer
	opt = tf.keras.optimizers.SGD(
		learning_rate=0.01, momentum=0.9, nesterov=False, name='SGD')
	run_name = ("resnet50" + '_' + "cifar100" + "_1")

	model(np.random.randn(batch_size, 3, 32, 32))
	train_data = train_data.take(min(len(train_data), num_iteration))
	model = tf.function(model)
	tf.profiler.experimental.start(logdir)
	step = 0
	for x, y in tqdm(train_data):
		with tf.profiler.experimental.Trace("inference", step_num=step):
			model(x)
		step += 1
	tf.profiler.experimental.stop()

if __name__ == "__main__":
	#the indexes where dense layer is faster than sparse
	mix_index = [23, 24, 26, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52]

	name = {0: "dense", 1: "maksed_dense",
		2: "dense_with_sparse_conv1x1", 3: "fully_sparse"}

	parser = argparse.ArgumentParser(
		description="profile resnet model",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--num_iteration", type=int, default=100)
	parser.add_argument("--directory", type=str, default="logdir")
	parser.add_argument("--mode", type=str, default="train")
	parser.add_argument("--mix_layers", type=bool, default=False)
	parser.add_argument("--profile_all", type=bool, default=False)
	args = parser.parse_args()
	if args.mix_layers:
		if args.mode == "train":
			profile_training(0, args.directory, num_iteration=args.num_iteration, mix_layers=True, mix_indexes=mix_index)
		else:
			profile_inference(i, "{}/inference/{}".format(args.directory, name[i]), args.num_iteration, mix_layers=True, mix_indexes=mix_index)
	elif args.profile_all:
		print("currently profiling mixed model")
		profile_training(3, "{}/train/{}".format(args.directory, "mixed"), args.num_iteration, mix_layers=True, mix_indexes=mix_index)
		profile_inference(3, "{}/inference/{}".format(args.directory, "mixed"), args.num_iteration, mix_layers=True, mix_indexes=mix_index)
		for i in range(4):
			print("currently profiling level {}".format(i))
			profile_training(i, "{}/train/{}".format(args.directory, name[i]), args.num_iteration)
			profile_inference(i, "{}/inference/{}".format(args.directory, name[i]), args.num_iteration)
	else:
		if args.mode == "train":
			profile_training(i, "{}/train/{}".format(args.directory, name[i]), args.num_iteration)
		if args.mode == "inference":
			profile_inference(i, "{}/inference/{}".format(args.directory, name[i]), args.num_iteration)
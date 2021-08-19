import os

from numpy.core.fromnumeric import size
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import sparse_matrix
from sgk.sparse import connectors
from sgk.sparse import ops
import numpy as np
import scipy.sparse as ss
from sparse_layers import SparseConv1x1,GeneralConv2D
from resnet import *


# tf.disable_v2_behavior()
# tf.enable_eager_execution()
sparsity = 0.5

m = 64
k = 64
n = 256
# a = sparse_matrix.SparseMatrix(
#           "kernel", [m, k],
#           connector=connectors.Uniform(sparsity),trainable=True)

# b = tf.constant(tf.random.uniform((k,n)))
# # print(b.trainable)

# with tf.GradientTape(persistent=True) as tape:
#   y = ops.spmm(a, b)
#   loss = tf.reduce_mean(y**2)
# grads_a = tape.gradient(loss, a._values)
# print(grads_a)
# grads_b = tape.gradient(loss, b)
# print(grads_b)

# # # with tf.Graph().as_default(), tf.Session() as sess,tf.GradientTape(persistent=True) as tape:
# c = GeneralConv2D(128,1, sparsity = 0.8)
# kernel_np = ss.random(128,64, density=0.5).toarray().astype(np.float32)
# d = tf.constant(np.random.uniform(size=(4,64,32,32)).astype(np.float32))
# c.build_from_np(kernel_np)
# with tf.GradientTape(persistent=True) as tape:
#   f = c(d)
#   loss = tf.reduce_mean(f)
# grad = tape.gradient(loss,c.trainable_weights)
# print("gradient")
# print(grad)
# print(c.trainable_weights)
# NCHW
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
b = tf.constant(tf.random.uniform((64,3,32,32)))
r = resnet50()
with tf.GradientTape(persistent=True) as tape:
  f = r(b)
  loss = tf.reduce_mean(f)
for var in r.trainable_variables:
  if("sparse_conv1x1" in var.name):
    print(var)
    break
grads = tape.gradient(loss,r.trainable_variables)
opt.apply_gradients(zip(grads, r.trainable_variables))

for grad,var in zip(grads, r.trainable_variables):
  if("sparse_conv1x1" in var.name):
    print(var)
    print(grad)
    break
# print(c.kernel.trainable_weights)
# for var in tf.trainable_variables():
#   print(var.name)
#   print(tape.gradient(loss,var))
#   # sess.run(loss)
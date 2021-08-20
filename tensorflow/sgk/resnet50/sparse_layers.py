# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom layers for sparse/dense inference."""
import tensorflow as tf

from sgk.sparse import connectors
from sgk.sparse import ops
import sparse_matrix
import scipy.sparse as ss
import numpy as np


class SparseConv2D(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 sparsity=0.7,
                 strides=(1, 1),
                 rates=(1, 1),
                 padding='VALID',
                 use_bias=False,
                 name="SparseConv2D"):
        super(SparseConv2D, self).__init__(name=name)
        self.filters = filters
        self.sparsity = sparsity
        self.use_bias = use_bias
        self.kernel_size = kernel_size
        self.initializer = tf.keras.initializers.GlorotUniform()
        if(isinstance(kernel_size, int)):
            self.kernel_size = (1, kernel_size, kernel_size, 1)
        elif (len(kernel_size) == 2):
            self.kernel_size = (1, kernel_size, kernel_size, 1)
        else:
            raise Exception("invalid format")

        if(isinstance(strides, int)):
            self.strides = (1, strides, strides, 1)
        elif (len(strides) == 2):
            self.strides = (1, strides[0], strides[1], 1)
        else:
            raise Exception("invalid format")

        self.padding = padding

        if(isinstance(rates, int)):
            self.rates = (1, rates, rates, 1)
        elif (len(rates) == 2):
            self.rates = (1, rates[0], rates[1], 1)
        else:
            raise Exception("invalid format")

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        input_channels = input_shape[1]
        self.kernel = sparse_matrix.SparseMatrix(
            self.name +
            "/kernel", [self.filters, input_channels *
                        self.kernel_size[1]*self.kernel_size[2]],
            connector=connectors.Uniform(self.sparsity), layer=self)
        self.kernel._values.assign(
            self.initializer(self.kernel._values.shape))
        if self.use_bias:
            self.bias = tf.Variable(self.initializer(
                shape=[self.filters], dtype=tf.float32), name=self.name+"/bias")
            # self.bias = tf.Variable(name="bias", validate_shape=[self.filters])

    def load_from_np(self, kernel_np):
        kernel_np = kernel_np.T
        assert kernel_np.shape[0] == self.filters, "numpy kernel dimension mismatch"
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.kernel = sparse_matrix.SparseMatrix(
            self.name + "/kernel", matrix=kernel_np, layer=self)

        self.sparsity = self.kernel.sparsity
        if(self.use_bias):
            self._trainable_weights.append(self.bias)

    def call(self, inputs, training=None):
        # assert len(input_shape) == 4, "expect 4 dimensional input"
        # NCHW -> NHWC
        flat_inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])

        flat_inputs = tf.image.extract_patches(
            flat_inputs, self.kernel_size, self.strides, self.rates, self.padding)

        flat_inputs = tf.transpose(flat_inputs, perm=[3, 0, 1, 2])

        # C NHW
        input_shape = flat_inputs.shape.as_list()
        flat_inputs = tf.reshape(flat_inputs, [flat_inputs.shape[0], -1])

        output_shape = [self.filters, input_shape[1],
                        input_shape[2], input_shape[3]]

        flat_output = ops.spmm(self.kernel, flat_inputs)
        out = tf.reshape(flat_output, output_shape)
        out = tf.transpose(out, perm=[1, 0, 2, 3])

        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias, data_format="NCHW")

        return out


class MaskedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, kernel_mask=None,
                 *args, **kwargs):
        super(MaskedConv2D, self).__init__(
            filters, kernel_size, *args, **kwargs)
        self.masked = False
        self.kernel_mask = kernel_mask

    def load_from_np(self, kernel_np):
        if (not self.built):
            raise Exception("Error: kernel not build")
        if (self.masked):
            raise Exception("Error: already masked")

        kernel_np = kernel_np.reshape(self.kernel.shape)
        self._underlying_kernel.assign(kernel_np)
        self.kernel_mask = np.copy(kernel_np)
        self.kernel_mask[self.kernel_mask != 0] = 1
        self.kernel_mask = tf.constant(self.kernel_mask)
        self._non_trainable_weights.append(self.kernel_mask)
        self.kernel = self._underlying_kernel * self.kernel_mask
        self.masked = True

    def call(self, input):
        if(self.masked):
            self.kernel = self._underlying_kernel * self.kernel_mask
        return super(MaskedConv2D, self).call(input)

    def build(self, input_shape):
        super(MaskedConv2D, self).build(input_shape)
        self._underlying_kernel = self.kernel
        if (self.kernel_mask is not None):
            self.kernel_mask = tf.constant(self.kernel_mask)
            self._non_trainable_weights.append(self.kernel_mask)
            self.kernel = self._underlying_kernel * self.kernel_mask
            self.masked = True


class SparseLinear(tf.keras.layers.Layer):
    """
    This is the sparse fully connected layer. 
    """

    def __init__(self, input_dim, output_dim, bias=True, sparsity=0.5):
        super(SparseLinear, self).__init__()
        self.sparsity = sparsity
        self.output_dim = output_dim
        self.w = sparse_matrix.SparseMatrix(
            self.name+"/weight", [output_dim, input_dim],
            connector=connectors.Uniform(self.sparsity), layer=self)
        self.b = None
        if bias:
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(initial_value=b_init(
                shape=(output_dim), dtype="float32"), name=self.name+"/bias", trainable=True)

    def call(self, inputs):
        # Only support [N, F_in]
        # Can not boardcast
        x = tf.transpose(inputs, [1, 0])
        out = ops.spmm(self.w, x)
        out = tf.transpose(out, [1, 0])
        if self.b is not None:
            return out + self.b
        return out

    def load_from_np(self, kernel_np):
        # assert kernel_np.shape[0] == self.filters, "numpy kernel dimension mismatch"
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.w = sparse_matrix.SparseMatrix(
            self.name+"/weight", matrix=kernel_np, layer=self)
        if(self.b is not None):
            self._trainable_weights.append(self.b)
        self.sparsity = self.w.sparsity


class SparseConv1x1(tf.keras.layers.Layer):
    """Sparse 1x1 convolution.

    NOTE: Only supports 1x1 convolutions, NCHW format, unit
    stride, and no padding.
    """

    def __init__(self,
                 filters,
                 sparsity,
                 use_bias=False,
                 activation=None,
                 name="SparseConv1x1"):
        super(SparseConv1x1, self).__init__(name=name)
        self.filters = filters
        self.sparsity = sparsity
        self.use_bias = use_bias
        self.activation = activation
        self.initializer = tf.keras.initializers.GlorotUniform()

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        input_channels = input_shape[1]
        self.kernel = sparse_matrix.SparseMatrix(
            self.name + "/kernel", [self.filters, input_channels],
            connector=connectors.Uniform(self.sparsity), layer=self)
        self.kernel._values.assign(self.initializer(self.kernel._values.shape))
        if self.use_bias:
            self.bias = tf.Variable(self.initializer(
                shape=[self.filters], dtype=tf.float32), name=self.name+"/bias")
            # self.bias = tf.Variable(name="bias", validate_shape=[self.filters])

    def load_from_np(self, kernel_np):
        kernel_np = kernel_np.T
        assert kernel_np.shape[0] == self.filters, "numpy kernel dimension mismatch"
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.kernel = sparse_matrix.SparseMatrix(
            self.name + "/kernel", matrix=kernel_np, layer=self)

        self.sparsity = self.kernel.sparsity
        if(self.use_bias):
            self._trainable_weights.append(self.bias)

    def call(self, inputs, training=None):
        input_shape = inputs.shape.as_list()
        assert len(input_shape) == 4, "expect 4 dimensional input"
        flat_inputs = tf.transpose(inputs, perm=[1, 0, 2, 3])
        flat_inputs = tf.reshape(
            inputs, [input_shape[1], input_shape[0]*input_shape[2] * input_shape[3]])

        output_shape = [self.filters, input_shape[0],
                        input_shape[2], input_shape[3]]

        # Use the fused kernel if possible.
        if self.use_bias and self.activation == tf.nn.relu:
            flat_output = ops.fused_spmm(self.kernel, flat_inputs, self.bias)
            return tf.reshape(flat_output, output_shape)
        flat_output = ops.spmm(self.kernel, flat_inputs)
        out = tf.reshape(flat_output, output_shape)

        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias, data_format="NCHW")
        if self.activation:
            out = self.activation(out)
        out = tf.transpose(out, perm=[1, 0, 2, 3])
        return out


class GeneralConv2D(tf.keras.layers.Layer):
    """
    Return sparse / dense conv2d, with paddings
    NCHW format
    """

    def __init__(self, filters, kernel_size, stride=1, padding=0,
                 bias=True, sparsity=0.5, name=None, sparse_level=3, save_input_shape=False):
        super(GeneralConv2D, self).__init__(name=name)
        self.padding_layer = None
        # Padding
        self.save_input_shape=save_input_shape
        if padding != 0:
            self.padding_layer = tf.keras.layers.ZeroPadding2D(
                padding=padding, data_format="channels_first")
        if sparse_level == 0:
            self.type = "level_{}_keras_layers_Conv2D_ksize_{}".format(sparse_level, kernel_size)
            self.conv_layer = tf.keras.layers.Conv2D(
                filters, kernel_size, use_bias=bias, strides=stride, name=name, data_format="channels_first")
        elif sparse_level == 1:
            self.type= "level_{}_MaskedConv2D_ksize_{}".format(sparse_level,kernel_size)
            self.conv_layer = MaskedConv2D(
                filters, kernel_size, use_bias=bias, strides=stride, name=name, data_format="channels_first")
        elif sparse_level == 2:
            if (kernel_size == 1 and stride == 1):
                self.type = "level_{}_SparseConv1x1_ksize_{}".format(sparse_level,kernel_size)
                self.conv_layer = SparseConv1x1(
                    filters, sparsity, use_bias=bias, name=name)
            else:
                self.type="level_{}_MaskedConv2D_ksize_{}".format(sparse_level,kernel_size)
                self.conv_layer = MaskedConv2D(
                    filters, kernel_size, use_bias=bias, strides=stride, name=name, data_format="channels_first")
        else:
            if (kernel_size == 1 and stride == 1):
                self.type="level_{}_SparseConv1x1_ksize_{}".format(sparse_level,kernel_size)
                self.conv_layer = SparseConv1x1(
                    filters, sparsity, use_bias=bias, name=name)
            else:
                self.type="level_{}_SparseConv2D_ksize_{}".format(sparse_level,kernel_size)
                self.conv_layer = SparseConv2D(
                    filters, kernel_size, use_bias=bias, strides=stride, name=name)

    def load_from_np(self, kernel_np):
        self.conv_layer.load_from_np(kernel_np)

    def call(self, inputs):
        if self.save_input_shape: self.layer_input_shape = inputs.shape
        if self.padding_layer is None:
            return self.conv_layer(inputs)
        inputs = self.padding_layer(inputs)
        return self.conv_layer(inputs)

    @property
    def kernel(self):
        return self.conv_layer.kernel

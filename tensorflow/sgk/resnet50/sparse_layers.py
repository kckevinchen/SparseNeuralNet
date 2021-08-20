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
# import tensorflow as tf

import tensorflow as tf

from sgk.sparse.ops.backend import kernels

import scipy.sparse as ss
import numpy as np


def _dense_to_sparse(matrix):
  """Converts dense numpy matrix to a csr sparse matrix."""
  assert len(matrix.shape) == 2

  # Extract the nonzero values.
  values = matrix.compress((matrix != 0).flatten())

  # Calculate the offset of each row.
  mask = (matrix != 0).astype(np.int32)
  row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))),
                               axis=0)

  # Create the row indices and sort them.
  row_indices = np.argsort(-1 * np.diff(row_offsets))

  # Extract the column indices for the nonzero values.
  x = mask * (np.arange(matrix.shape[1]) + 1)
  column_indices = x.compress((x != 0).flatten())
  column_indices = column_indices - 1

  # Cast the desired precision.
  values = values.astype(np.float32)
  row_indices, row_offsets, column_indices = [
      x.astype(np.uint32) for x in
      [row_indices, row_offsets, column_indices]
  ]
  return values, row_indices, row_offsets, column_indices

def _to_sparse_matrix(kernel_np, self,reinit=False):
        values_, row_indices_, row_offsets_, column_indices_ = _dense_to_sparse(
        kernel_np)
        if(reinit):
            values_ = self.initializer(
                shape=[values_.shape], dtype=tf.float32)
        self._nnz = np.count_nonzero(kernel_np)
        self._rows = tf.Variable(
            kernel_np.shape[0],
            trainable=False,
            name=self._name + "_rows",
            dtype=tf.int32)
        self._columns = tf.Variable(
            kernel_np.shape[1],
            trainable=False,
            name=self._name + "_columns",
            dtype=tf.int32)

        # Convert the sparse matrix to TensorFlow variables.
        self._values = tf.Variable(
            values_,
            trainable=True,
            name=self._name + "_values",
            dtype=self._dtype)
        self._row_indices = tf.Variable(
            row_indices_,
            trainable=False,
            name=self._name + "_row_indices",
            dtype=tf.uint32)
        self._row_offsets = tf.Variable(
            row_offsets_,
            trainable=False,
            name=self._name + "_row_offsets",
            dtype=tf.uint32)
        self._column_indices = tf.Variable(
            column_indices_,
            trainable=False,
            name=self._name + "_column_indices",
            dtype=tf.uint32)



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
        if(self.built):
            return
        input_channels = input_shape[1]
        kernel_np = ss.random(self.filters,input_channels*self.kernel_size[1]*self.kernel_size[2], density=1-self.sparsity).toarray()
        self.build_from_np(kernel_np,reinit= True)


    def build_from_np(self, kernel_np,reinit = False):
        # kernel_np = kernel_np.T
        if(kernel_np.shape[1] == self.filters):
            kernel_np = kernel_np.T
        else:
            assert kernel_np.shape[0] == self.filters , "numpy kernel dimension mismatch"

        assert self.built == False, "Already build"
        _to_sparse_matrix(kernel_np,self,reinit)
        if self.use_bias:
            self.bias = tf.Variable(self.initializer(
                shape=[self.filters], dtype=tf.float32), name=self.name+"/bias")
        self.built = True


    def call(self, inputs, training=None):
        # assert len(input_shape) == 4, "expect 4 dimensional input"
        # NCHW -> NHWC
        flat_inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])

        flat_inputs = tf.image.extract_patches(
            flat_inputs, self.kernel_size, self.strides, self.rates, self.padding)

        flat_inputs = tf.transpose(flat_inputs, perm=[3, 0, 1, 2])

        # C NHW
        input_shape = flat_inputs.shape
        flat_inputs = tf.reshape(flat_inputs, [flat_inputs.shape[0], -1])

        output_shape = [self.filters, input_shape[1],
                        input_shape[2], input_shape[3]]


        # SPMM
        flat_output = kernels.spmm(self._rows, self._columns,
                      self._values, self._row_indices,
                      self._row_offsets, self._column_indices,
                      flat_inputs, False,False)
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
        self.kernel_np = None

    def build_from_np(self, kernel_np):
        if (self.built):
            raise Exception("Error: kernel built")
        if (self.masked):
            raise Exception("Error: already masked")
        if(kernel_np.shape[0] == self.filters):
            self.kernel_np = kernel_np
        elif(kernel_np.shape[1] == self.filters):
            self.kernel_np = kernel_np.T
        else:
            raise Exception("Error: kernel size")
        self.kernel_np = self.kernel_np.reshape((self.kernel_size[0],self.kernel_size[1],-1,self.filters))
        initializer = tf.constant_initializer(self.kernel_np)
        self.kernel_initializer = initializer
        # Build until run time

    def call(self, input):
        if(self.masked):
            self.kernel = self._underlying_kernel * self.kernel_mask
        return super(MaskedConv2D, self).call(input)

    def build(self, input_shape):
        # NCHW
        if(self.built):
            return
        super(MaskedConv2D, self).build(input_shape)
        self._underlying_kernel = self.kernel
        if(self.kernel_np is not None):
            #  N*in,out
            self.kernel_mask = np.copy(self.kernel_np)
            self.kernel_mask[self.kernel_mask != 0] = 1


        if (self.kernel_mask is not None):
            self.kernel_mask = tf.constant(self.kernel_mask)
            self._non_trainable_weights.append(self.kernel_mask)
            self.kernel = self._underlying_kernel * self.kernel_mask
            self.masked = True
       


class SparseLinear(tf.keras.layers.Layer):
    """
    This is the sparse fully connected layer. 
    """

    def __init__(self, output_dim, use_bias=True, sparsity=0.5):
        super(SparseLinear, self).__init__()
        self.sparsity = sparsity
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.initializer = tf.keras.initializers.GlorotUniform()
    def build(self,input_shape):
        if(self.built):
            return
        input_dim = input_shape[0]
        kernel_np = ss.random(self.output_dim,input_dim, density=1-self.sparsity).toarray()
        self.build_from_np(kernel_np)

    def call(self, inputs):
        # Only support [N, F_in]
        # Can not boardcast
        x = tf.transpose(inputs, [1, 0])

        out = kernels.spmm(self._rows, self._columns,
                      self._values, self._row_indices,
                      self._row_offsets, self._column_indices,
                      x, False,False)
        out = tf.transpose(out, [1, 0])
        if self.use_bias:
            return out + self.bias
        return out
    def build_from_np(self, kernel_np,reinit = False):
        if(kernel_np.shape[1] == self.output_dim):
            kernel_np = kernel_np.T
        else:
            assert kernel_np.shape[0] == self.filters , "numpy kernel dimension mismatch"
        assert self.built == False, "Already build"
        _to_sparse_matrix(kernel_np,self,reinit)
        if self.use_bias:
            self.bias = tf.Variable(self.initializer(
                shape=[self.output_dim], dtype=tf.float32), name=self.name+"/bias")
        self.built = True


class SparseConv1x1(tf.keras.layers.Layer):
    """Sparse 1x1 convolution.

    NOTE: Only supports 1x1 convolutions, NCHW format, unit
    stride, and no padding.
    """

    def __init__(self,
                 filters,
                 sparsity = 0.5,
                 use_bias=False,
                 name="SparseConv1x1"):
        super(SparseConv1x1, self).__init__(name=name)
        self.filters = filters
        self.sparsity = sparsity
        self.use_bias = use_bias
        self.initializer = tf.keras.initializers.glorot_uniform()

    def build(self, input_shape):
        if(self.built):
            return
        input_channels = input_shape[1]
        kernel_np = ss.random(self.filters,input_channels, density=1-self.sparsity).toarray()
        self.build_from_np(kernel_np,reinit= True)


    def build_from_np(self, kernel_np,reinit = False):
        # kernel_np = kernel_np.T
        if(kernel_np.shape[1] == self.filters):
            kernel_np = kernel_np.T
        else:
            assert kernel_np.shape[0] == self.filters , "numpy kernel dimension mismatch"
        assert self.built == False, "Already build"
        _to_sparse_matrix(kernel_np,self,reinit)
        if self.use_bias:
            self.bias = tf.Variable(self.initializer(
                shape=[self.filters], dtype=tf.float32), name=self.name+"/bias")
        self.built = True

    def call(self, inputs):
        input_shape = inputs.shape
        assert len(input_shape) == 4, "expect 4 dimensional input"
        flat_inputs = tf.transpose(inputs, perm=[1, 0, 2, 3])
        flat_inputs = tf.reshape(
            flat_inputs, [input_shape[1], input_shape[0]*input_shape[2] * input_shape[3]])

        output_shape = [self.filters, input_shape[0],
                        input_shape[2], input_shape[3]]

        flat_output = kernels.spmm(self._rows, self._columns,
                      self._values, self._row_indices,
                      self._row_offsets, self._column_indices,
                      flat_inputs, False,False)
        out = tf.reshape(flat_output, output_shape)
        out = tf.transpose(out, perm=[1, 0, 2, 3])
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias, data_format="NCHW")
        return out


class GeneralConv2D(tf.keras.layers.Layer):
    """
    Return sparse / dense conv2d, with paddings
    NCHW format
    """

    def __init__(self, filters, kernel_size, stride=1, padding=0,
                 bias=True, sparsity=0.5, name=None, sparse_level=3):
        super(GeneralConv2D, self).__init__(name=name)
        self.padding_layer = None
        # Padding
        if padding != 0:
            self.padding_layer = tf.keras.layers.ZeroPadding2D(
                padding=padding, data_format="channels_first")
        if sparse_level == 0:
            self.type = "level_{}_keras_layers_Conv2D_ksize_{}".format(
                sparse_level, kernel_size)
            self.conv_layer = tf.keras.layers.Conv2D(
                filters, kernel_size, use_bias=bias, strides=stride, name=name, data_format="channels_first")
        elif sparse_level == 1:
            self.type = "level_{}_MaskedConv2D_ksize_{}".format(
                sparse_level, kernel_size)
            self.conv_layer = MaskedConv2D(
                filters, kernel_size, use_bias=bias, strides=stride, name=name, data_format="channels_first")
        elif sparse_level == 2:
            if (kernel_size == 1 and stride == 1):
                self.type = "level_{}_SparseConv1x1_ksize_{}".format(
                    sparse_level, kernel_size)
                self.conv_layer = SparseConv1x1(
                    filters, sparsity, use_bias=bias, name=name)
            else:
                self.type = "level_{}_MaskedConv2D_ksize_{}".format(
                    sparse_level, kernel_size)
                self.conv_layer = MaskedConv2D(
                    filters, kernel_size, use_bias=bias, strides=stride, name=name, data_format="channels_first")
        else:
            if (kernel_size == 1 and stride == 1):
                self.type = "level_{}_SparseConv1x1_ksize_{}".format(
                    sparse_level, kernel_size)
                self.conv_layer = SparseConv1x1(
                    filters, sparsity, use_bias=bias, name=name)
            else:
                self.type = "level_{}_SparseConv2D_ksize_{}".format(
                    sparse_level, kernel_size)
                self.conv_layer = SparseConv2D(
                    filters, kernel_size, use_bias=bias, strides=stride, name=name)

    def build_from_np(self, kernel_np):
        self.conv_layer.build_from_np(kernel_np)

    def call(self, inputs):
        if self.padding_layer is None:
            return self.conv_layer(inputs)
        inputs = self.padding_layer(inputs)
        return self.conv_layer(inputs)
    @property
    def nnz(self):
        if(isinstance(self.conv_layer,MaskedConv2D)):
            return tf.math.count_nonzero(self.conv_layer._underlying_kernel,dtype=tf.dtypes.int32)
        else:
            return self.conv_layer._nnz
    @property
    def size(self):
        if(isinstance(self.conv_layer,MaskedConv2D)):
            return tf.size(self.conv_layer._underlying_kernel,out_type=tf.dtypes.int32)
        else:
            return self.conv_layer._rows *self.conv_layer._columns




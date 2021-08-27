
import tensorflow as tf
import numpy as np
from resnet import *
import os
import scipy.io as sio
from sparse_matrix import *

def compute_mean_var(image):
    # image.shape: [image_num, w, h, c]
    mean = []
    var = []
    for c in range(image.shape[-1]):
        mean.append(np.mean(image[:, :, :, c]))
        var.append(np.std(image[:, :, :, c]))
    return mean, var


def norm_images(image):
    # image.shape: [image_num, w, h, c]
    image = image.astype('float32')
    mean, var = compute_mean_var(image)
    image[:, :, :, 0] = (image[:, :, :, 0] - mean[0]) / var[0]
    image[:, :, :, 1] = (image[:, :, :, 1] - mean[1]) / var[1]
    image[:, :, :, 2] = (image[:, :, :, 2] - mean[2]) / var[2]
    return image


def preprocess_data(x):
    paddings = [[0, 0], [4, 4], [4, 4], [0, 0]]
    x = tf.pad(x, paddings)
    x = tf.image.random_crop(x, [x.shape[0], 32, 32, x.shape[3]])
    x = tf.image.random_flip_left_right(x).numpy()
    return norm_images(x)



def flatten_layers(model):
    edge_layer = []
    if(hasattr(model, "layers")):
        for i in model.layers:
            edge_layer += flatten_layers(i)
    else:
        edge_layer += [model]
    return edge_layer


def restore_from_mtx(model, mtx_dir):
    edge_layers = flatten_layers(model)
    prunable_layers = filter(
        lambda layer: isinstance(layer, GeneralConv2D) or isinstance(
            layer, SparseLinear), edge_layers)
    mtx_files = sorted(os.listdir(mtx_dir), key=lambda x: int(x.split("_")[0]))
    for layer, mtx_f in zip(prunable_layers, mtx_files):
        matrix = np.array(sio.mmread(os.path.join(
            mtx_dir, mtx_f)).todense()).astype(np.float32)
        layer.build_from_np(matrix.T)

def mix_dense_sparse(model, dense_index):
    edge_layers = flatten_layers(model)
    conv_layers = list(filter(
        lambda layer: isinstance(layer, GeneralConv2D), edge_layers
    ))
    for i in dense_index:
        conv_layers[i].change_to_dense()


def global_sparsity(model):
    edge_layers = flatten_layers(model)
    prunable_layers = filter(
        lambda layer: isinstance(layer, GeneralConv2D) or isinstance(
            layer, SparseLinear), edge_layers)
    nnz = 0
    size = 0
    for layer in prunable_layers:
        if(isinstance(layer, GeneralConv2D)):
            size += layer.size
            nnz += layer.nnz
        else:
            size += layer._rows*layer._columns
            nnz += layer._nnz
    return round((nnz/size).numpy(),2)

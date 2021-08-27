from read_profile import *
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from profile_layer import get_layers_conv1x1, get_layers_conv3x3
"""
logdir/train/conv1x1/layer_1/dense
"""

def loop_dir(root_dir, fname="kernel_stats.pb"):
    acc = []
    for subdir, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(fname):
                acc.append(os.path.join(subdir, f))
    return acc

#layer basic info ==========================
def _get_layer_info(layers, indexes):
    """
    return a dictionary of
    {layer_number: (sparsity, #input_entries, #weight_entries)}
    """
    result = {}
    for i in indexes:
        l = layers[i][1]
        result[i] = (1-(l.nnz / l.size).numpy() , reduce(lambda x, y: x*y, l.layer_input_shape), l.size.numpy())
    return result


def get_layer_info():
    conv1x1_info = _get_layer_info(*get_layers_conv1x1())
    conv3x3_info = _get_layer_info(*get_layers_conv3x3())
    return { **conv1x1_info, **conv3x3_info }

# layer runtime classify ====================
# def get_spmm_other_runtime(dictionary):
#     """
#     given the dictionary, return 
#     """



#layer runtime plotter =============================
def process_layer_path(path):
    """
    layer path is of the form
    root_dir / {train / inference} / {conv1x1 / conv3x3} / layer_{ i } / {dense / masked_dense / sparse}
    """
    splits = path.split("/")
    root_dir, mode_type, conv_type, layer_num, layer_type = splits[:5]
    return mode_type, conv_type, layer_num, layer_type 

def make_layer_dictionary(root_dir):
    result = {}
    file_paths = loop_dir(root_dir)
    for path in file_paths:
        k = process_layer_path(path)
        result[k] = read_kernel_stat_pb(path)
    return result

def filter_layer_dictionary(mode_type, conv_type, layer_type, layer_dict):
    result = {}
    for k, v in layer_dict.items():
        cur_mode_type, cur_conv_type, cur_layer_num, cur_layer_type = k
        if cur_mode_type == mode_type and cur_conv_type == conv_type and cur_layer_type == layer_type:
            result[k] = v
    return result

def get_total_time(op_time_arr):
    """
    Given the op_time array [(op_name, time_in_ns)],
    return the total time used
    """
    acc = 0
    for i in op_time_arr:
        acc += i[1]
    return acc

def average_runtime_by_type(layer_dictionary):
    conv_types = ["conv1x1", "conv3x3"]
    layer_types = ["dense", "masked_dense", "sparse"]
    mode_types = ["train", "inference"]
    tmp, result = {}, {}
    for k, v in layer_dictionary.items():
        mode_type, conv_type, _, layer_type = k
        name = "{}_{}_{}".format(mode_type, conv_type, layer_type)
        time = get_total_time(v)
        if name not in tmp:
            tmp[name] = [time]
        else:
            tmp[name].append(time)
    for k, v in tmp.items():
        result[k] = np.average(v)
    return result

def plot_layer_average_runtime(log_directory, plot_directory):
    tmp = make_layer_dictionary(log_directory)
    runtime_dictionary = average_runtime_by_type(tmp)
    conv_types = ["conv1x1", "conv3x3"]
    layer_types = ["dense", "masked_dense", "sparse"]
    mode_types = ["train", "inference"]
    indexes = []
    for c in conv_types:
        for l in layer_types:
            indexes.append("{}_{}".format(c, l))
    train, inference = [], [] 
    for index in indexes:
        train_k, inference_k = "train_{}".format(index), "inference_{}".format(index)
        train.append(int(runtime_dictionary[train_k]))
        inference.append(int(runtime_dictionary[inference_k]))
    df = pd.DataFrame({"train": train, "inference":inference}, index=indexes)
    # plt.figure(figsize=(16,20), dpi=150)
    df.plot.bar()
    plt.xlabel("layer name")
    plt.ylabel("runtime (ns)")
    plt.title("average layer runtime in Resnet50")
    plt.xticks(rotation=15)
    plt.savefig("{}/layer_runtime.pdf".format("plots"),bbox_inches="tight")

# model runtime plotter =====================================
def process_model_path(path):
    splits = path.split("/")
    root_dir, mode_type, model_type = splits[:3]
    return mode_type, model_type

def make_model_dictionary(root_dir):
    result = {}
    paths = loop_dir(root_dir, "overview_page.pb")
    for path in paths:
        result["{}_{}".format(*process_model_path(path))] = read_overview_page(path)
    return result

def plot_model_runtime(log_directory, plot_directory):
    mode_types = ["inference, train"]
    model_types = ["dense", "masked_dense", "dense_with_sparse_conv1x1", "fully_sparse"]
    indexes = []
    for mode in mode_types:
        for model in model_types:
            indexes.append("{}_{}".format())

def get_op_name(op_time_arr):
    """
    Given an array of (op_name, run_time), return all the
    op's name in set
    """
    op_time_arr = set([(i[0], i[2]) for i in op_time_arr])
    return op_time_arr

def main():
    # plot_layer_average_runtime("profiler_logs", "plots")


    layer_dict = make_layer_dictionary("profiler_logs")
    print(len(layer_dict))
    exit()
    train_conv1x1_dense = filter_layer_dictionary("train", "conv1x1", "dense", layer_dict)
    _, op_time_arr = list(train_conv1x1_dense.items())[0]
    op_time_arr = get_op_name(op_time_arr)
    num = 0
    acc = set()
    for k, v in train_conv1x1_dense.items():
        v = get_op_name(v)
        if v != op_time_arr:
            num += 1
        acc = acc.union(v)
    print("acc length {}, op_time_arr length {}, num is {}".format(len(acc), len(op_time_arr), num))

    for i in acc:
        print(i[0], "\n=====\n",i[1], "\n\n\n------------\n\n\n")
        # exit()
if __name__ == "__main__":
    main()
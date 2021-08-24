from read_profile import *
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
logdir/train/conv1x1/layer_1/dense
"""

def loop_dir(root_dir):
    acc = []
    for subdir, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith("kernel_stats.pb"):
                acc.append(os.path.join(subdir, f))
    return acc

def process_layer_path(path):
    """
    layer path is of the form
    root_dir / {train / inference} / {conv1x1 / conv3x3} / layer_{ i } / {dense / masked_dense / sparse}
    """
    splits = path.split("/")
    root_dir, mode_type, conv_type, layer_num, layer_type = splits[:5]
    return root_dir, mode_type, conv_type, layer_num, layer_type 

def make_dictionary(root_dir):
    result = {}
    file_paths = loop_dir(root_dir)
    for path in file_paths:
        k = process_layer_path(path)
        result[k] = read_kernel_stat_pb(path)
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
        _, mode_type, conv_type, _, layer_type = k
        name = "{}_{}_{}".format(mode_type, conv_type, layer_type)
        time = get_total_time(v)
        if name not in tmp:
            tmp[name] = [time]
        else:
            tmp[name].append(time)
    for k, v in tmp.items():
        result[k] = np.average(v)
    return result

def plot_average_runtime(runtime_dictionary):
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
    plt.savefig("plots/layer_runtime.pdf",bbox_inches="tight")


def main():
    tmp = make_dictionary("profiler_logs")
    result = average_runtime_by_type(tmp)
    plot_average_runtime(result)
if __name__ == "__main__":
    main()
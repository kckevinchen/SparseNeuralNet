import tf_stats_pb2
from google.protobuf.json_format import MessageToDict, MessageToJson

"""
In order to use this file, 
	1 compile the ~/tensorflow/tensorflow/core/profiler/protobuf/tf_stats.proto
	Here is the code: protoc -I. --python_out . tf_stats.proto 
	More on: https://developers.google.com/protocol-buffers/docs/pythontutorial#compiling-your-protocol-buffers

	2 make sure the protobuf is installed in pip

	3 Make the tracefiles through the tf profiler

	4 read the xxx.tensorflow_stats.pb
"""

def read_pb(tracefile):
    fd = open(tracefile, 'rb')
    A = tf_stats_pb2.TfStatsDatabase()
    A.ParseFromString(fd.read())
    fd.close()

    decoded = MessageToDict(A)
    print(decoded)

if __name__ == "__main__":
	directory = "/home/tian/utea/SparseBenchmark/exp/SparseNeuralNet/tensorflow/sgk/resnet50/profiler_logs/conv1x1/dense/plugins/profile/2021_08_20_12_22_20/2a85d1cda857.tensorflow_stats.pb"
	read_matmul_timing(directory)
	
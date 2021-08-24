import kernel_stats_pb2
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

def read_tf_stat_pb(tracefile):
	fd = open(tracefile, 'rb')
	A = tf_stats_pb2.TfStatsDatabase()
	A.ParseFromString(fd.read())
	fd.close()

	decoded = MessageToDict(A)
	print(decoded)


def read_kernel_stat_pb(tracefile):
	fd = open(tracefile, 'rb')
	A = kernel_stats_pb2.KernelStatsDb()
	A.ParseFromString(fd.read())
	fd.close()
	decoded = MessageToDict(A)
	return process_reports(decoded)

def process_reports(reports):
	"""
	process the decoded report, return [(kernel_name, avg_runtime_ns), ...]
	"""
	reports = reports["reports"]
	result = []
	for r in reports:
		name = r["name"]
		total_duration = int(r["totalDurationNs"])
		num_iteration = int(r["occurrences"])
		result.append((name, int(total_duration / num_iteration)))
	return result

if __name__ == "__main__":
	directory = "profiler_logs/inference/conv1x1/layer_1/dense/plugins/profile/2021_08_23_18_49_48/af663c433fab.kernel_stats.pb"
	reports = read_kernel_stat_pb(directory)
	print(reports)
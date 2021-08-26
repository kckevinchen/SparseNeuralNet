from tensorboard_plugin_profile.protobuf import kernel_stats_pb2
from tensorboard_plugin_profile.protobuf import tf_stats_pb2
from tensorboard_plugin_profile.protobuf import overview_page_pb2
from google.protobuf.json_format import MessageToDict, MessageToJson
"""
 
"""
substitution = {
	#conv3x3
	"void sputnik::(anonymous namespace)::CudaSddmmKernel<float4, 4, 32, 32, 8, 0>(int, int, int, int const*, int const*, int const*, float const*, float const*, float*)": "SPMM",
	"void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReshapingOp<Eigen::DSizes<int, 4> const, Eigen::TensorImagePatchOp<-1l, -1l, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReshapingOp<Eigen::DSizes<int, 4> const, Eigen::TensorImagePatchOp<-1l, -1l, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int)" : "img2col",
}



def read_tf_stat_pb(tracefile):
	fd = open(tracefile, 'rb')
	A = tf_stats_pb2.TfStatsDatabase()
	A.ParseFromString(fd.read())
	fd.close()

	decoded = MessageToDict(A)
	return decoded


def read_kernel_stat_pb(tracefile):
	fd = open(tracefile, 'rb')
	A = kernel_stats_pb2.KernelStatsDb()
	A.ParseFromString(fd.read())
	fd.close()
	decoded = MessageToDict(A)
	return process_reports(decoded)

def read_overview_page(tracefile):
	fd = open(tracefile, 'rb')
	A = overview_page_pb2.OverviewPage()
	A.ParseFromString(fd.read())
	fd.close()
	decoded = MessageToDict(A)
	step_time = decoded["inputAnalysis"]["stepTimeSummary"]["average"]
	compute_percent = decoded["inputAnalysis"]["computePercent"]
	return step_time, compute_percent

def process_op_name(op_name):
	op_name = op_name.split("/")
	return op_name[-1]

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
		# print((name, int(total_duration / num_iteration), process_op_name(r["opName"])))
		# exit()
		result.append((name, int(total_duration / num_iteration), process_op_name(r["opName"])))
	return result





if __name__ == "__main__":
	path = "/home/tian/utea/SparseBenchmark/exp/SparseNeuralNet/tensorflow/sgk/resnet50/profiler_logs/train/conv3x3/layer_48/sparse/plugins/profile/2021_08_23_18_51_27/af663c433fab.kernel_stats.pb"
	read_kernel_stat_pb(path)
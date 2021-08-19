import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as ss
import math
import torch.nn.init as init
import torch.nn.functional as F


class SparseConv2D(torch.nn.Module):
    """Sparse 2d convolution.

    NOTE: Only supports NCHW format and no padding.
    """
    __constants__ = ['in_channels', 'out_channels']
    in_channels: int
    out_channels: int
    weight: torch.Tensor

    def __init__(self, in_channels, out_channels,kernel_size,stride = 1,padding = 0,bias=True):
        super(SparseConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if(isinstance(kernel_size, tuple)):
            self.kernel_h, self.kernel_w = kernel_size
        elif(isinstance(kernel_size, int)):
            self.kernel_h, self.kernel_w = (kernel_size,kernel_size)
        else:
            print("Unsupported kernel size")
            exit()

        if(isinstance(stride, tuple)):
            self.stride_h, self.stride_w = stride
        elif(isinstance(stride, int)):
            self.stride_h, self.stride_w = (stride,stride)
        else:
            print("Unsupported stride")
            exit()

        if(isinstance(padding, tuple)):
            self.padding_h, self.padding_w = padding
        elif(isinstance(padding, int)):
            self.padding_h, self.padding_w = (padding,padding)
        else:
            print("Unsupported padding")
            exit()

        if (bias):
            self.bias = torch.nn.Parameter(torch.empty(out_channels,1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self,sparsity = 0,weight_np = None ):
        if weight_np is None :
            weight_np = ss.random(self.out_channels,self.in_channels*self.kernel_w*self.kernel_h, density=1-sparsity).toarray()
        weight_np = weight_np.astype(np.float32)
        if (weight_np.ndim == 4):
            weight_np = weight_np.reshape(weight_np.shape[0],-1)
        assert weight_np.shape == (self.out_channels,self.in_channels*self.kernel_w*self.kernel_h), "Wrong matrix size for kernel"
        self.weight = torch.nn.Parameter(torch.from_numpy(weight_np).to_sparse().coalesce())
        self.build_bias()

    def build_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        flat_x,out_shape = self.pad_img2col(x)
        if not self.bias is None:
            flat_output = torch.sparse.addmm(self.bias,self.weight, flat_x)
        else:
            flat_output = torch.sparse.mm(self.weight, flat_x)
        out =   flat_output.reshape([self.out_channels , *out_shape]).transpose(0,1)
        return out

    def pad_img2col(self,x):
        if(self.padding_w > 0 or self.padding_h > 0):
            x = F.pad(x, (self.padding_h,self.padding_w),"constant", 0)  
        # NCHW -> C*K*K, NHW
        input_windows = x.unfold(2, self.kernel_h, self.stride_h).unfold(3, self.kernel_w, self.stride_w)
        # C,k*k,N, H, W
        input_windows = input_windows.contiguous().view(*input_windows.size()[:-2], -1).permute(1,4,0,2,3)
        out_shape = [input_windows.shape[2],input_windows.shape[3],input_windows.shape[4]]
        input_windows = input_windows.reshape(input_windows.shape[0]*input_windows.shape[1],input_windows.shape[2]*input_windows.shape[3]*input_windows.shape[4]).contiguous()
        return input_windows,out_shape


class SparseConv1x1(torch.nn.Module):
    """Sparse 1x1 convolution.

    NOTE: Only supports 1x1 convolutions, NCHW format, unit
    stride, and no padding.
    """
    __constants__ = ['in_channels', 'out_channels']
    in_channels: int
    out_channels: int
    weight: torch.Tensor

    def __init__(self, in_channels, out_channels,
            bias=True):
        super(SparseConv1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if (bias):
            self.bias = torch.nn.Parameter(torch.empty(out_channels,1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self,sparsity = 0,weight_np = None ):
        if weight_np is None :
            weight_np = ss.random(self.out_channels,self.in_channels, density=1-sparsity).toarray()
        weight_np = weight_np.astype(np.float32)
        self.weight = torch.nn.Parameter(torch.from_numpy(weight_np).to_sparse().coalesce())
        self.build_bias()

    def build_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
      # input NCHW
        input_shape = x.shape
        output_shape = [self.out_channels,input_shape[0], input_shape[2], input_shape[3]]
        flat_x = x.transpose(0,1).reshape([input_shape[1],input_shape[0]*input_shape[2] * input_shape[3]]).contiguous()
        if not self.bias is None:
            flat_output = torch.sparse.addmm(self.bias,self.weight, flat_x)
        else:
            flat_output = torch.sparse.mm(self.weight, flat_x)
        out =   flat_output.reshape(output_shape).transpose(0,1)
        return out

class SparseLinear(torch.nn.Module):
    """Sparse linear layer.

    NOTE: (N ,*, H_in) format
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    def __init__(self,in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if (bias):
            self.bias = torch.nn.Parameter( torch.empty(out_features,1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self,sparsity = 0,weight_np = None ):
        if weight_np is None :
            weight_np = ss.random(self.out_features,self.in_features, density=1-sparsity).toarray()
        weight_np = weight_np.astype(np.float32)
        self.weight = torch.nn.Parameter(torch.from_numpy(weight_np).to_sparse().coalesce())
        self.build_bias()

    def build_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self,x):
        input_shape = list(x.shape)
        inner_shape = input_shape[1:-1]
        batch_size = input_shape[0]
        in_features = input_shape[-1]
        flat_x = x.transpose(0,-1).reshape([in_features, -1])
        if not self.bias is None:
            output = torch.sparse.addmm(self.bias,self.weight, flat_x)
        else:
            output = torch.sparse.mm(self.weight, flat_x)
        output = output.reshape([self.out_features, *inner_shape,batch_size]).transpose(0,-1)
        return output
    

def conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0,  bias=True):
  if (kernel_size == 1 and stride ==1  and padding == 0):
    return SparseConv1x1(in_channels, out_channels,bias)
  else:
    return SparseConv2D(in_channels, out_channels, kernel_size, stride, padding)

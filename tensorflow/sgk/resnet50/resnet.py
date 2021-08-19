import tensorflow as tf
from resnet_block import *

class Resnet(tf.keras.Model):
  def __init__(self, block, num_blocks, num_classes=10, temp=1.0, in_planes=64, stable_resnet=False):
    super(Resnet, self).__init__()
    self.in_planes = in_planes
    if stable_resnet:
      L=0
      for x in num_blocks:
        L+=x
      self.L = L
    else:
      self.L = 1

    self.masks = None

    self.conv1 = GeneralConv2D(self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-5)
    self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)
    self.avgpool = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first")
    self.fc = SparseLinear(in_planes*8*block.expansion, num_classes)
    self.temp = temp
    self.flatten = tf.keras.layers.Flatten(data_format="channels_first")
  
  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride, self.L))
      self.in_planes = planes * block.expansion
    return tf.keras.Sequential([*layers])
  
  def call(self, x):
    out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)
    out = self.flatten(out)
    out = self.fc(out) / self.temp
    return out

def resnet18(temp=1, **kwargs):
  model = Resnet(BasicBlock, [2,2,2,2], temp=temp, **kwargs)
  return model

def resnet50(temp=1, **kwargs):
  model = Resnet(Bottleneck, [3,4,6,3], temp=temp, **kwargs)
  return model


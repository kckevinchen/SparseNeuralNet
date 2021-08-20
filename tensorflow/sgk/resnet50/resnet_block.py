import tensorflow as tf
from  sparse_layers import *
'''
Layers for ResNet50 model
'''
#I tried my best to match Kevin's interface and network structure to create a fair
#comparison
class BasicBlock(tf.keras.Model):
  expansion=1
  
  def __init__(self, in_planes, planes, stride=1, L=1,sparse_level = 3, save_input_shape=False):
    super(BasicBlock, self).__init__()
    self.conv1 = GeneralConv2D(planes, kernel_size=3, stride=stride, padding=1, bias=False,sparse_level=sparse_level, save_input_shape=save_input_shape)
    self.bn1 = tf.keras.layers.BatchNormalization(axis=1)
    self.conv2 = GeneralConv2D(planes, kernel_size=3, stride=1, padding=1, bias=False,sparse_level=sparse_level, save_input_shape=save_input_shape)
    self.bn2 = tf.keras.layers.BatchNormalization(axis=1)
    self.factor=L**(-0.5)
    self.shortcut = tf.keras.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = tf.keras.Sequential([
          GeneralConv2D(self.expansion * planes, kernel_size=1, stride=stride, bias=False, save_input_shape=save_input_shape),
          tf.keras.layers.BatchNormalization(axis=1)
      ])
  def call(self, x):
    out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out = out * self.factor + self.shortcut(x)
    out = tf.keras.activations.relu(out)
    return out



class Bottleneck(tf.keras.Model):
  expansion=4

  def __init__(self, in_planes, planes, stride=1, L=1,sparse_level = 3, save_input_shape=False):
    super(Bottleneck, self).__init__()
    self.conv1 = GeneralConv2D( planes, kernel_size=1, bias=False,sparse_level=sparse_level, save_input_shape=save_input_shape)
    self.bn1 = tf.keras.layers.BatchNormalization(axis=1)
    self.conv2 = GeneralConv2D( planes, kernel_size=3, stride=stride, padding=1, bias=False,sparse_level=sparse_level, save_input_shape=save_input_shape)
    self.bn2 =  tf.keras.layers.BatchNormalization(axis=1)
    self.conv3 = GeneralConv2D( self.expansion * planes, kernel_size=1, bias=False,sparse_level=sparse_level, save_input_shape=save_input_shape)
    self.bn3 = tf.keras.layers.BatchNormalization(axis=1)
    self.factor = L**(-0.05)

    self.shortcut = tf.keras.Sequential()
    if stride!=1 or in_planes != self.expansion* planes:
      self.shortcut = tf.keras.Sequential([
          GeneralConv2D(self.expansion * planes, kernel_size=1, stride=stride, bias=False,sparse_level=sparse_level, save_input_shape=save_input_shape),
          tf.keras.layers.BatchNormalization(axis=1)
      ])
  def call(self, x):
    out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
    out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out = out * self.factor + self.shortcut(x)
    out = tf.keras.activations.relu(out)
    return out


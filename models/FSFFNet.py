# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from functools import reduce
import tensorflow as tf
from tensorflow import keras
import numpy as np
#### Custom function for conv2d: conv_block
def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
  
  if(conv_type == 'ds'):
    x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
  else:
    x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)  
  
  x = tf.keras.layers.BatchNormalization()(x)
  
  if (relu):
    #x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Activation(lambda x: tf.nn.swish(x)) (x)
  
  return x

#### residual custom method
def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    
    
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

"""#### Bottleneck custom method ###"""

def bottleneck_block(inputs, filters, kernel, t, strides, n):
  x = _res_bottleneck(inputs, filters, kernel, t, strides)
  
  for i in range(1, n):
    x = _res_bottleneck(x, filters, kernel, t, 1, True)

  return x    

"""### Feature Scaling Module ###"""
def FSM(inputs):

    dilation_rates = np.array([(6, 6), (12, 12), (18, 18)])
    shape = list(inputs.shape)
    
    # Dilation branch 1
    conv3x3_d6 = tf.keras.layers.SeparableConv2D(
        40, 3, strides=1, padding='same', dilation_rate=dilation_rates[0], name="ASPP_sep_conv3x3_d6")(inputs)
    norm3x3_d6 = tf.keras.layers.BatchNormalization(
        name="ASPP_Sep_conv3x3_d6_batch_norm")(conv3x3_d6)
    relu3x3_d6 = tf.keras.layers.Activation(
        'relu', name="ASPP_Sep_conv3x3_d6_relu")(norm3x3_d6)

    #Dilation branch 2
    conv3x3_d12 = tf.keras.layers.SeparableConv2D(
        40, 3, strides=1, padding='same', dilation_rate=dilation_rates[1], name="ASPP_sep_conv3x3_d12")(inputs)
    norm3x3_d12 = tf.keras.layers.BatchNormalization(
        name="ASPP_Sep_conv3x3_d12_batch_norm")(conv3x3_d12)
    relu3x3_d12 = tf.keras.layers.Activation(
        'relu', name="ASPP_Sep_conv3x3_d12_relu")(norm3x3_d12)

    #Dilation branch 3
    conv3x3_d18 = tf.keras.layers.SeparableConv2D(
        40, 3, strides=1, padding='same', dilation_rate=dilation_rates[2], name="ASPP_sep_conv3x3_d18")(inputs)
    norm3x3_d18 = tf.keras.layers.BatchNormalization(
        name="ASPP_Sep_conv3x3_d18_batch_norm")(conv3x3_d18)
    relu3x3_d18 = tf.keras.layers.Activation(
        'relu', name="ASPP_Sep_conv3x3_d18_relu")(norm3x3_d18)
  
    # Feature pooling branch
    pool = tf.keras.layers.AveragePooling2D(pool_size=(
        shape[1], shape[2]), name="ASPP_Ave_Pool")(inputs)
    conv1 = tf.keras.layers.Conv2D(
        40, 1, strides=1, padding='same', use_bias=False, name='ASPP_conv1')(pool)
    norm1 = tf.keras.layers.BatchNormalization(
        name='ASPP_conv1_batch_norm')(conv1)
    relu1 = tf.keras.layers.Activation('relu', name='ASPP_conv1_relu')(norm1)
    upsampling = tf.keras.layers.UpSampling2D(
        size=(shape[1], shape[2]), interpolation='bilinear')(relu1)

    # Concatenate all the above layers
    concat = tf.keras.layers.Concatenate(name='ASPP_concatenate')(
        [relu3x3_d6, relu3x3_d12, relu3x3_d18, upsampling])

    # Return the result
    return concat

MOMENTUM = 0.997
EPSILON = 1e-4
def SeparableConvBlock(num_channels, kernel_size, strides, freeze_bn=False):
    f1 = tf.keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, dilation_rate=2,)
    
    f2 = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='')
    #f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def model(num_classes=19, input_size=(1024, 2048, 3)):

  # Input Layer
  input_layer = tf.keras.layers.Input(shape=input_size, name = 'input_layer')

  ## Step 1: Learning to DownSample
  C1 = conv_block(input_layer, 'conv', 32, (3, 3), strides = (2, 2))
  
  C2 = bottleneck_block(C1, 24, (3, 3), t=1, strides=2, n=1)
  C2 = bottleneck_block(C2, 32, (3, 3), t=6, strides=1, n=2)
  
  C3 = bottleneck_block(C2, 48, (3, 3), t=6, strides=2, n=3)  
  C3 = bottleneck_block(C3, 64, (3, 3), t=6, strides=1, n=2)

  C4 = bottleneck_block(C3, 96, (3, 3), t=6, strides=2, n=3)  
  C4 = bottleneck_block(C4, 128, (3, 3), t=6, strides=1, n=2)

  C5 = bottleneck_block(C4, 160, (3, 3), t=6, strides=2, n=1)

  """### Intermediate stage ###"""
  #Featue Scaling Module
  C5 = FSM(C5)

  P5_in = C5
  P5_in_1 = tf.keras.layers.Conv2D(64, 1, 1, padding='same', activation=None, name='resample_p6/conv2d')(P5_in)
  P5_in_1 = tf.keras.layers.BatchNormalization()(P5_in_1)
  
  #Creating 6th stage feature map
  P6_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P5_in)
  
  #Creating 6th stage feature map
  P7_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)

  """### Feature Fusion Module ###"""
  P3_in = C3
  P3_in_1 = tf.keras.layers.Conv2D(64, 1, 1, padding='same', activation=None, name = f'fpn_cells/cell/fnode3/conv2d')(P3_in)
  P3_in_1 = tf.keras.layers.BatchNormalization()(P3_in_1)

  P4_in = C4
  P4_in_1 = tf.keras.layers.Conv2D(64, 1, 1, padding='same', activation=None, name = f'fpn_cells/cell/fnode2/conv2d')(P4_in)
  P4_in_1 = tf.keras.layers.BatchNormalization()(P4_in_1)
  
  # Top-down path
  P7_U = tf.keras.layers.UpSampling2D()(P7_in)

  P6_td = tf.keras.layers.Add()([P6_in, P7_U])
  P6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
  P6_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P6_td)
  P6_U = tf.keras.layers.UpSampling2D()(P6_td)

  P5_td = tf.keras.layers.Add()([P5_in_1, P6_U])
  P5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
  P5_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P5_td)
  P5_U = tf.keras.layers.UpSampling2D()(P5_td)

  P4_td = tf.keras.layers.Add()([P4_in_1, P5_U])
  P4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
  P4_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P4_td)
  P4_U = tf.keras.layers.UpSampling2D()(P4_td)

  P3_out = tf.keras.layers.Add()([P3_in_1, P4_U])
  P3_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
  P3_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P3_out)

  #Bottom-Up path
  P3_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
  P4_out = tf.keras.layers.Add()([P4_td, P3_D])
  P4_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
  P4_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P4_out)
  P4_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)

  P5_out = tf.keras.layers.Add()([P5_td, P4_D])
  P5_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
  P5_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P5_out)
  P5_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)

  P6_out = tf.keras.layers.Add()([P6_td, P5_D])
  P6_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
  P6_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P6_out)
  P6_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)

  P7_out = tf.keras.layers.Add()([P7_in, P6_D])
  P7_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
  P7_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P7_out)

  #Final top-down path
  P7_final_U = tf.keras.layers.UpSampling2D()(P7_out)

  P6_U = tf.keras.layers.Add()([P6_in, P7_final_U, P6_out]) 
  P6_U = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_U)
  P6_U = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P6_U)
  P6_U = tf.keras.layers.UpSampling2D()(P6_U)

  P5_U = tf.keras.layers.Add()([P5_in_1, P5_out, P6_U])
  P5_U = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_U)
  P5_U = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P5_U)
  P5_U = tf.keras.layers.UpSampling2D()(P5_U)

  P4_U = tf.keras.layers.Add()([P4_in_1, P4_out, P5_U])
  P4_U = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P4_U)
  P4_U = tf.keras.layers.UpSampling2D()(P4_U)

  P3_U = tf.keras.layers.Add()([P3_in, P3_out, P4_U]) 
  P3_U = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_U)
  P3_U = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P3_U)

  output = tf.keras.layers.UpSampling2D((2,2))(P3_U)
  output = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(2,2), padding='same')(output)
  output = tf.keras.layers.BatchNormalization()(output)
  output = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(output)

  #Fusining with local feature at 2nd stage
  C2 = tf.keras.layers.Conv2D(64, 1, 1, padding='same', activation=None, name='C2_conv')(C2)
  C2 = tf.keras.layers.BatchNormalization()(C2)

  output = tf.keras.layers.Add()([C2, output])
  output = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(output) 
  
  output = tf.keras.layers.UpSampling2D((2,2))(output)
  output = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(2,2), padding='same')(output)
  output = tf.keras.layers.BatchNormalization()(output)
  output = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(output)
  output = tf.keras.layers.Conv2D(48, kernel_size=(1,1), strides=(1,1), padding='same')(output)

  #Fusining with local feature at 1st stage
  C1 = tf.keras.layers.Conv2D(48, 1, 1, padding='same', activation=None, name='C1_conv')(C1)
  C1 = tf.keras.layers.BatchNormalization()(C1)

  output = tf.keras.layers.Add()([C1, output])
  output = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(output)

  """### Step 4: Classifier ###"""
  classifier = tf.keras.layers.SeparableConv2D(48, (3, 3), padding='same', strides = (1, 1), dilation_rate=2, name = 'DSConv1_classifier')(output)
  classifier = tf.keras.layers.BatchNormalization()(classifier)
  #classifier = tf.keras.activations.relu(classifier)
  classifier = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(classifier)

  classifier = tf.keras.layers.SeparableConv2D(32, (3, 3), padding='same', strides = (1, 1), dilation_rate=2, name = 'DSConv2_classifier')(classifier)
  classifier = tf.keras.layers.BatchNormalization()(classifier)
  #classifier = tf.keras.activations.relu(classifier)
  classifier = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(classifier)

  
  classifier = tf.keras.layers.Conv2D(num_classes, 1, 1, padding='same', activation=None,
                                      kernel_regularizer=keras.regularizers.l2(0.00004),
                                      bias_regularizer=keras.regularizers.l2(0.00004))(classifier)
  
  classifier = tf.keras.layers.Dropout(0.35)(classifier)

  classifier = tf.keras.layers.UpSampling2D((2, 2))(classifier)
  
  #Since its likely that mixed precision training is used, make sure softmax is float32
  classifier = tf.dtypes.cast(classifier, tf.float32)
  classifier = tf.keras.activations.softmax(classifier)

  FSFFNet = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'FSFFNet')

  return FSFFNet

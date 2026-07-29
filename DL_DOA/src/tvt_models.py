# ==========================================================================
# TVT Model Architectures - UNet and ResNet
# ---------------------------------------------------------------
# This module defines two neural network architectures:
# 
# 1. UNet:
#    - Encoder-decoder convolutional architecture with skip connections.
#    - Designed for dense regression tasks, such as blob detection.
#    - Configurable depth, width, and optional autoencoder-style latent layer.
#
# 2. ResNet:
#    - Deep residual convolutional network for super-resolution.
#    - Employs skip connections for stable training across 64 stacked layers.
#
# Both models are used for AoA/AoD inference in mmWave MIMO scenarios.
# ==========================================================================

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Input, BatchNormalization, MaxPooling2D, Flatten, Activation, Add, Conv2DTranspose
from tensorflow.keras import Model

# ==============================
# UNet Building Blocks
# ==============================

# Applies a convolutional layer followed by batch normalization and ReLU
def Conv_Block(inputs, model_width, kernel, multiplier):
    x = tf.keras.layers.Conv2D(model_width * multiplier, kernel, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Applies a transposed convolution (upsampling) followed by BN and ReLU
def trans_conv2D(inputs, model_width, multiplier):
    x = tf.keras.layers.Conv2DTranspose(model_width * multiplier, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Concatenates multiple tensors along the channel axis
def Concat_Block(input1, *argv):
    cat = input1
    for arg in argv:
        cat = tf.keras.layers.concatenate([cat, arg], axis=-1)
    return cat

# Upsampling block (alternative to Conv2DTranspose)
def upConv_Block(inputs, size=(2, 2)):
    up = tf.keras.layers.UpSampling2D(size=size)(inputs)
    return up

# Optional autoencoder-style feature bottleneck for latent vector encoding
def Feature_Extraction_Block(inputs, model_width, feature_number):
    shape = inputs.shape
    latent = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(feature_number, name='features')(latent)
    latent = tf.keras.layers.Dense(model_width * shape[1] * shape[2])(latent)
    latent = tf.keras.layers.Reshape((shape[1], shape[2], model_width))(latent)
    return latent


# ==============================
# UNet Model Constructor
# ==============================
def UNet(length=64, width=64, model_depth=5, num_channel=2, model_width=32, kernel_size=3, 
         problem_type='Regression', output_nums=1, ds=0, ae=0, ag=0, lstm=0, 
         feature_number=1024, is_transconv=True, M=256):
    
    """
    Constructs a flexible UNet architecture.

    Parameters:
        length, width        : Spatial input dimensions
        model_depth          : Number of encoder/decoder layers
        num_channel          : Number of input channels
        model_width          : Base number of filters
        kernel_size          : Convolution kernel size
        ae                   : If 1, include autoencoder latent bottleneck
        feature_number       : Latent feature size if ae=1
        M                    : Controls final output upsampling (256 or 512)
    
    Returns:
        tf.keras.Model       : A compiled UNet model
    """
    if length == 0 or model_depth == 0 or model_width == 0 or num_channel == 0 or kernel_size == 0:
        raise ValueError("Please Check the Values of the Input Parameters!")

    convs = {}
    levels = []

    inputs = tf.keras.Input((length, width, num_channel))
    pool = inputs

    # Encoder path
    for i in range(1, (model_depth + 1)):
        conv = Conv_Block(pool, model_width, kernel_size, 2 ** (i - 1))
        conv = Conv_Block(conv, model_width, kernel_size, 2 ** (i - 1))
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
        convs["conv%s" % i] = conv

    # Optional autoencoder latent vector
    if ae == 1:
        pool = Feature_Extraction_Block(pool, model_width, feature_number)

    # Bottom block (deepest layer)
    conv = Conv_Block(pool, model_width, kernel_size, 2 ** model_depth)
    conv = Conv_Block(conv, model_width, kernel_size, 2 ** model_depth)

    deconv = conv
    convs_list = list(convs.values())

    # Decoder path with skip connections
    for j in range(0, model_depth):
        skip_connection = convs_list[model_depth - j - 1]
        deconv = trans_conv2D(deconv, model_width, 2 ** (model_depth - j - 1))
        deconv = Concat_Block(deconv, skip_connection)
        deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** (model_depth - j - 1))
        deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** (model_depth - j - 1))

     # Final upsampling and refinement
    deconv = tf.keras.layers.Conv2DTranspose(16 , (2, 2), strides=(2, 2), padding='same')(deconv)
    h = tf.keras.layers.Conv2DTranspose(16 , (2, 2), strides=(2, 2), padding='same')(inputs)

    deconv = Concat_Block(deconv, h)
    deconv = Conv_Block(deconv, model_width, kernel_size, 16)

    # Output layer
    if M == 256:
        outputs = tf.keras.layers.Conv2DTranspose(1 , (2, 2), activation='linear', strides=(2, 2), padding='same')(deconv)
    elif M == 512:
        outputs = tf.keras.layers.Conv2DTranspose(1 , (4, 4), strides=(4, 4), padding='same')(deconv)
    
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

# ==============================
# Residual Block for ResNet
# ==============================

# from tensorflow.keras.layers import Dense, Conv2D, Input, BatchNormalization, MaxPooling2D, Flatten, Activation, Add, Conv2DTranspose
# from tensorflow.keras import Model

# Define model architecture
def res_conv(x, filters=12):
    """
    Defines a basic residual block with two Conv2D layers.

    Parameters:
        x        : Input tensor
        filters  : Number of filters for both layers

    Returns:
        x        : Output tensor after residual connection
    """

    x_skip = x
    f1 =  filters
    # first block
    x = Conv2D(f1, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Conv2D(f1, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    # add 
    x = Add()([x, x_skip])
    x = tf.keras.layers.Activation("relu")(x)

    return x

# ==============================
# ResNet Model Constructor
# ==============================
def Resnet(input_shape=(64,64,2), output_dim = 1):
    """
    Constructs a deep ResNet model for super-resolution inference.

    Parameters:
        input_shape  : Shape of the input tensor
        output_dim   : Number of output channels (typically 1)

    Returns:
        model        : A tf.keras.Model with 64 residual layers
    """

    x_in = Input(shape=input_shape)

    # Initial upsampling layer
    x = tf.keras.layers.Conv2DTranspose(12, (5,5), strides=(2, 2), padding='same',output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None)(x_in)

    # Residual learning stack
    for i in range(64):
        x = res_conv(x)

    # Final upsampling layer
    x = tf.keras.layers.Conv2DTranspose(output_dim, (5,5), strides=(2, 2), padding='same',output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None)(x)

    model = Model(inputs=x_in, outputs=x)

    return model
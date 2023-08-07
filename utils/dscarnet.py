import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.layers import MaxPool2D, GlobalMaxPool2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Concatenate, Dense,AveragePooling2D
from tensorflow.keras.backend import int_shape

def Conv2D_BN(x,filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              batchnorm=False):
    if batchnorm:
        x = Conv2D(
          filters, (num_row, num_col),
          strides=strides,
          padding=padding,
          use_bias=False, )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        x = Conv2D(
          filters, (num_row, num_col),
          strides=strides,
          padding=padding,
          use_bias=False, 
          )(x)
        x = Activation('relu')(x)
    return x

def conv2d_bn(x,filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              batchnorm=False):
    lamda = 0.01
    if batchnorm:
        x = Conv2D(
          filters, (num_row, num_col),
          strides=strides,
          padding=padding,
          use_bias=False, )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        x = Conv2D(
          filters, (num_row, num_col),
          strides=strides,
          padding=padding,
          use_bias=False, 
          kernel_regularizer=tf.keras.regularizers.l2(lamda),
          bias_regularizer=tf.keras.regularizers.l2(lamda), 
          activity_regularizer=tf.keras.regularizers.l2(lamda)
          )(x)
        x = Activation('relu')(x)
    return x


def single_dscarnet(input_shape, filter_number = 64,
               n_outputs = 2, 
               conv1_kernel_size = 19,
               n_inception = 1,
               dense_layers = [128], 
               dense_avf = 'relu', batchnorm=False,
               last_avf = 'softmax'):

    tf.keras.backend.clear_session()
    assert len(input_shape) == 3
    
    
    inputs = Input(input_shape)
    
    x = Conv2D_BN(inputs, 64, conv1_kernel_size, conv1_kernel_size, 'same', 1, batchnorm=batchnorm)
    if int_shape(x)[1]>25:
        x = Conv2D_BN(x, 96, 5, 5, 'valid', 2, batchnorm=batchnorm)
        x = Conv2D_BN(x, 128, 5, 5,  'valid', 1, batchnorm=batchnorm)
        
    else:
        x = Conv2D_BN(x, 96, 5, 5,  'valid', 1, batchnorm=batchnorm)
        x = Conv2D_BN(x, 128, 5, 5,  'same', 1, batchnorm=batchnorm)

    x = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)

    for i in range(n_inception):
        
        unit = 48*(2**i)
        branch1x1 = conv2d_bn(x, unit, 1, 1,batchnorm=batchnorm)
        branch5x5 = conv2d_bn(x, 0.5*unit, 1, 1,batchnorm=batchnorm)
        branch5x5 = conv2d_bn(branch5x5, unit, 5, 5,batchnorm=batchnorm)
        branch3x3dbl = conv2d_bn(x, unit, 1, 1,batchnorm=batchnorm)
        branch3x3dbl = conv2d_bn(branch3x3dbl, unit*2, 3, 3,batchnorm=batchnorm)
        branch3x3dbl = conv2d_bn(branch3x3dbl, unit*2, 3, 3,batchnorm=batchnorm)
        branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, unit, 1, 1,batchnorm=batchnorm)
        x = Concatenate()([branch1x1, branch5x5, branch3x3dbl, branch_pool])

    x = GlobalMaxPool2D()(x)
    for units in dense_layers:

        x = Dense(units, activation = dense_avf)(x)


    outputs = Dense(n_outputs,activation=last_avf)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model    


def dual_dscarnet(input_shape, input_shape2, filter_number = 64,
               n_outputs = 2, 
               conv1_kernel_size = 19,
               batchnorm = False,
               n_inception = 1,
               dense_layers = [128], 
               dense_avf = 'relu', 
               last_avf = 'softmax'):

    tf.keras.backend.clear_session()
    assert len(input_shape) == 3
    
    inputs = Input(input_shape)
    
    x = Conv2D_BN(inputs, 64, conv1_kernel_size, conv1_kernel_size, 'same', 1, batchnorm=batchnorm)
    if int_shape(x)[1]>25:
        x = Conv2D_BN(x, 96, 5, 5, 'valid', 2, batchnorm=batchnorm)
        x = Conv2D_BN(x, 128, 5, 5,  'valid', 1, batchnorm=batchnorm)
        
    else:
        x = Conv2D_BN(x, 96, 5, 5,  'valid', 1, batchnorm=batchnorm)
        x = Conv2D_BN(x, 128, 5, 5,  'same', 1, batchnorm=batchnorm)

    x = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)

    for i in range(n_inception):
        
        unit = 48*(2**i)
        branch1x1 = conv2d_bn(x, unit, 1, 1,batchnorm=batchnorm)
        branch5x5 = conv2d_bn(x, 0.5*unit, 1, 1,batchnorm=batchnorm)
        branch5x5 = conv2d_bn(branch5x5, unit, 5, 5,batchnorm=batchnorm)
        branch3x3dbl = conv2d_bn(x, unit, 1, 1,batchnorm=batchnorm)
        branch3x3dbl = conv2d_bn(branch3x3dbl, unit*2, 3, 3,batchnorm=batchnorm)
        branch3x3dbl = conv2d_bn(branch3x3dbl, unit*2, 3, 3,batchnorm=batchnorm)
        branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, unit, 1, 1,batchnorm=batchnorm)
        x = Concatenate()([branch1x1, branch5x5, branch3x3dbl, branch_pool])

    x = GlobalMaxPool2D()(x)
    
    inputs2 = Input(input_shape2)
    
    x2 = Conv2D_BN(inputs2, 64, conv1_kernel_size, conv1_kernel_size, 'same', 1, batchnorm=batchnorm)
    if int_shape(x)[1]>25:
        x2 = Conv2D_BN(x2, 96, 5, 5, 'valid', 2, batchnorm=batchnorm)
        x2 = Conv2D_BN(x2, 128, 5, 5,  'valid', 1, batchnorm=batchnorm)
        
    else:
        x2 = Conv2D_BN(x2, 96, 5, 5,  'valid', 1, batchnorm=batchnorm)
        x2 = Conv2D_BN(x2, 128, 5, 5,  'same', 1, batchnorm=batchnorm)

    x2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x2)

    for i in range(n_inception):
        
        unit = 48*(2**i)
        branch1x1_2 = conv2d_bn(x2, unit, 1, 1,batchnorm=batchnorm)
        branch5x5_2 = conv2d_bn(x2, 0.5*unit, 1, 1,batchnorm=batchnorm)
        branch5x5_2 = conv2d_bn(branch5x5_2, unit, 5, 5,batchnorm=batchnorm)
        branch3x3dbl_2 = conv2d_bn(x2, unit, 1, 1,batchnorm=batchnorm)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl_2, unit*2, 3, 3,batchnorm=batchnorm)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl_2, unit*2, 3, 3,batchnorm=batchnorm)
        branch_pool_2 = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x2)
        branch_pool_2 = conv2d_bn(branch_pool_2, unit, 1, 1,batchnorm=batchnorm)
        x2 = Concatenate()([branch1x1_2, branch5x5_2, branch3x3dbl_2, branch_pool_2])

    x2 = GlobalMaxPool2D()(x2)
    
    
    x = Concatenate()([x, x2]) 

    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)

    outputs = Dense(n_outputs,activation=last_avf)(x)
    
    model = tf.keras.Model(inputs=[inputs,inputs2], outputs=outputs)
    
    return model





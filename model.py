from keras.layers import Input, Conv2D, MaxPooling2D, Add, Activation, Multiply, Dense, multiply, \
    Lambda, concatenate, Concatenate, Reshape, Dropout, BatchNormalization, Flatten
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
import math


def attention_residual_block(x, num_filters, stride):
    skip = x
    x = Conv2D(num_filters, kernel_size=(1, 1), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), strides=(stride, stride), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters * 3, kernel_size=(1, 1), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = channel_attention(x)
    x = spatial_attention(x)

    skip = Conv2D(num_filters * 3, kernel_size=(1, 1), strides=(stride, stride), padding='same')(
        skip)
    x = Add()([x, skip])
    x = Activation('relu')(x)
    return x


def spatial_attention(input_tensor):
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_tensor)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_tensor)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    attention_map = Conv2D(1, (5, 5), strides=(1, 1), padding='same', activation='sigmoid')(concat)
    output_tensor = Multiply()([input_tensor, attention_map])
    return output_tensor


def channel_attention(input_tensor, ratio=4):
    channel = K.int_shape(input_tensor)[-1]
    avg_pool = Lambda(lambda x: K.mean(x, axis=[1, 2]))(input_tensor)
    max_pool = Lambda(lambda x: K.max(x, axis=[1, 2]))(input_tensor)
    concat = concatenate([avg_pool, max_pool])
    concat = Dense(channel // ratio, activation='relu')(concat)
    concat = Dense(channel, activation='sigmoid')(concat)
    return multiply([input_tensor, concat])


def pyramid_pooling(input_tensor, pool_sizes):
    pooled_outputs = []
    input_shape = input_tensor.shape.as_list()
    height, width = input_shape[1], input_shape[2]

    for size in pool_sizes:
        pool_height = math.ceil(height / size)
        pool_width = math.ceil(width / size)
        pooled = MaxPooling2D(pool_size=(pool_height, pool_width), strides=(pool_height, pool_width), padding='same')(
            input_tensor)
        b_height, b_width, channel_num = pooled.shape[1], pooled.shape[2], pooled.shape[3]
        pooled_flattened = Reshape((b_height * b_width, channel_num))(pooled)
        pooled_outputs.append(pooled_flattened)

    return Concatenate(axis=1)(pooled_outputs)


def attention_resnet(input_shape=(None, None, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(0.01))(inputs)
    print(f"x:{x.shape}")
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    print(f"x:{x.shape}")
    for _ in range(2):
        x = attention_residual_block(x, 32, stride=1)

    print(f"x:{x.shape}")
    x = attention_residual_block(x, 64, stride=2)
    for _ in range(1):
        x = attention_residual_block(x, 64, stride=1)

    print(f"x:{x.shape}")
    x = attention_residual_block(x, 128, stride=2)

    print(f"x:{x.shape}")
    x = pyramid_pooling(x, [1, 2, 3])
    x = Flatten()(x)

    print(f"x:{x.shape}")
    print(f"x.shape[1]:{x.shape[1]}")
    for _ in range(2):
        x = Dense(int(x.shape[1]) // 2, activation='relu')(x)
        x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

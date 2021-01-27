from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, concatenate, UpSampling2D
from keras.optimizers import Adam


def WNet(input_size_rgb=(512, 512, 3), input_size_ir=(512, 512, 1),  classNum=2, learning_rate=3e-4):
    inputs_rgb = Input(input_size_rgb)
    inputs_ir = Input(input_size_ir)
    '''rgb编码部分'''
    # rgb encoder
    # stage 1
    conv1_rgb = BatchNormalization()(
        Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs_rgb))
    conv1_rgb = BatchNormalization()(
        Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_rgb))
    #  对于空间数据的最大池化
    pool1_rgb = MaxPooling2D(pool_size=(2, 2))(conv1_rgb)
    # stage 2
    conv2_rgb = BatchNormalization()(
        Conv2D(96, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_rgb))
    conv2_rgb = BatchNormalization()(
        Conv2D(96, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_rgb))
    pool2_rgb = MaxPooling2D(pool_size=(2, 2))(conv2_rgb)
    # stage 3
    conv3_rgb = BatchNormalization()(
        Conv2D(192, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_rgb))
    conv3_rgb = BatchNormalization()(
        Conv2D(192, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_rgb))
    pool3_rgb = MaxPooling2D(pool_size=(2, 2))(conv3_rgb)
    # stage 4
    conv4_rgb = BatchNormalization()(
        Conv2D(384, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_rgb))
    conv4_rgb = BatchNormalization()(
        Conv2D(384, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_rgb))
    pool4_rgb = MaxPooling2D(pool_size=(2, 2))(conv4_rgb)
    # stage 5
    conv5_rgb = BatchNormalization()(
        Conv2D(768, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_rgb))
    '''ir编码部分'''
    # ir encoder
    # stage 1
    conv1_ir = BatchNormalization()(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs_ir))
    conv1_ir = BatchNormalization()(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_ir))
    #  对于空间数据的最大池化
    pool1_ir = MaxPooling2D(pool_size=(2, 2))(conv1_ir)
    # stage 2
    conv2_ir = BatchNormalization()(
        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_ir))
    conv2_ir = BatchNormalization()(
        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_ir))
    pool2_ir = MaxPooling2D(pool_size=(2, 2))(conv2_ir)
    # stage 3
    conv3_ir = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_ir))
    conv3_ir = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_ir))
    pool3_ir = MaxPooling2D(pool_size=(2, 2))(conv3_ir)
    # stage 4
    conv4_ir = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_ir))
    conv4_ir = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_ir))
    pool4_ir = MaxPooling2D(pool_size=(2, 2))(conv4_ir)
    # stage 5
    conv5_ir = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_ir))
    '''解码部分'''
    # decoder
    # stage 5
    x = concatenate([conv5_ir, conv5_rgb], axis=3)
    conv5 = BatchNormalization()(
        Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x))
    # stage 4
    up4 = UpSampling2D(size=(2, 2))(conv5)
    cat4 = concatenate([conv4_ir, conv4_rgb], axis=3)
    cat4 = concatenate([cat4, up4], axis=3)
    conv4 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cat4))
    conv4 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4))
    # stage 3
    up3 = UpSampling2D(size=(2, 2))(conv4)
    cat3 = concatenate([conv3_ir, conv3_rgb], axis=3)
    cat3 = concatenate([cat3, up3], axis=3)
    conv3 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cat3))
    conv3 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3))
    # stage 2
    up2 = UpSampling2D(size=(2, 2))(conv3)
    cat2 = concatenate([conv2_ir, conv2_rgb], axis=3)
    cat2 = concatenate([cat2, up2], axis=3)
    conv2 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cat2))
    conv2 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2))
    # stage 1
    up1 = UpSampling2D(size=(2, 2))(conv2)
    conv1 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up1))
    conv1 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1))
    conv_out = Conv2D(classNum, 1, activation='softmax')(conv1)
    '''完善模型'''
    model = Model(inputs=[inputs_rgb, inputs_ir], outputs=conv_out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['categorical_accuracy'])
    return model

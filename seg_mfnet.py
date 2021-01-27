from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Add
from keras.optimizers import Adam


def miniInception(inputs=None , channel=96):
    conv_left = BatchNormalization()(
        Conv2D(channel//2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs))
    conv_right = BatchNormalization()(
        Conv2D(channel//2, 3, activation='relu', padding='same', dilation_rate=2, kernel_initializer='he_normal')(inputs))
    x = concatenate([conv_left, conv_right], axis=3)
    return x


def MFNet(pretrained_weights=None, input_size_rgb=(512, 512, 3), input_size_ir=(512, 512, 1),  classNum=2, learning_rate=3e-4):
    inputs_rgb = Input(input_size_rgb)
    inputs_ir = Input(input_size_ir)
    # rgb encoder
    # stage 1
    conv1_rgb = BatchNormalization()(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs_rgb))
    #  对于空间数据的最大池化
    pool1_rgb = MaxPooling2D(pool_size=(2, 2))(conv1_rgb)
    # stage 2
    conv2_rgb = BatchNormalization()(
        Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_rgb))
    conv2_rgb = BatchNormalization()(
        Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_rgb))
    pool2_rgb = MaxPooling2D(pool_size=(2, 2))(conv2_rgb)
    # stage 3
    conv3_rgb = BatchNormalization()(
        Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_rgb))
    conv3_rgb = BatchNormalization()(
        Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_rgb))
    pool3_rgb = MaxPooling2D(pool_size=(2, 2))(conv3_rgb)
    # stage 4
    conv4_rgb = miniInception(inputs=pool3_rgb)
    conv4_rgb = miniInception(inputs=conv4_rgb)
    conv4_rgb = miniInception(inputs=conv4_rgb)
    pool4_rgb = MaxPooling2D(pool_size=(2, 2))(conv4_rgb)
    # stage 5
    conv5_rgb = miniInception(inputs=pool4_rgb)
    conv5_rgb = miniInception(inputs=conv5_rgb)
    conv5_rgb = miniInception(inputs=conv5_rgb)

    # ir encoder
    # stage 1
    conv1_ir = BatchNormalization()(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs_ir))
    #  对于空间数据的最大池化
    pool1_ir = MaxPooling2D(pool_size=(2, 2))(conv1_ir)
    # stage 2
    conv2_ir = BatchNormalization()(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_ir))
    conv2_ir = BatchNormalization()(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_ir))
    pool2_ir = MaxPooling2D(pool_size=(2, 2))(conv2_ir)
    # stage 3
    conv3_ir = BatchNormalization()(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_ir))
    conv3_ir = BatchNormalization()(
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_ir))
    pool3_ir = MaxPooling2D(pool_size=(2, 2))(conv3_ir)
    # stage 4
    conv4_ir = miniInception(inputs=pool3_ir, channel=32)
    conv4_ir = miniInception(inputs=conv4_ir, channel=32)
    conv4_ir = miniInception(inputs=conv4_ir, channel=32)
    pool4_ir = MaxPooling2D(pool_size=(2, 2))(conv4_ir)
    # stage 5
    conv5_ir = miniInception(inputs=pool4_ir, channel=32)
    conv5_ir = miniInception(inputs=conv5_ir, channel=32)
    conv5_ir = miniInception(inputs=conv5_ir, channel=32)

    # decoder
    # stage 5
    x = concatenate([conv5_ir, conv5_rgb], axis=3)
    # stage 4
    up4 = UpSampling2D(size=(2, 2))(x)
    cat4 = concatenate([conv4_ir, conv4_rgb], axis=3)
    shortcut4 = Add()([cat4, up4])
    conv4 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(shortcut4))
    # stage 3
    up3 = UpSampling2D(size=(2, 2))(conv4)
    cat3 = concatenate([conv3_ir, conv3_rgb], axis=3)
    shortcut3 = Add()([cat3, up3])
    conv3 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(shortcut3))
    # stage 2
    up2 = UpSampling2D(size=(2, 2))(conv3)
    cat2 = concatenate([conv2_ir, conv2_rgb], axis=3)
    shortcut2 = Add()([cat2, up2])
    conv2 = BatchNormalization()(
        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(shortcut2))
    # stage 1
    up1 = UpSampling2D(size=(2, 2))(conv2)
    conv1 = BatchNormalization()(
        Conv2D(9, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up1))
    conv_out = Conv2D(classNum, 1, activation='softmax')(conv1)
    model = Model(inputs=[inputs_rgb, inputs_ir], outputs=conv_out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['categorical_accuracy'])
    #  如果有预训练的权重
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

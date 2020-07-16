from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from data import load_data
from utils import process_output
from skimage.io import imsave
import numpy as np


def _unet(input_shape=(512, 512, 3)):
    inputs = Input(shape=input_shape, name="input")

    # Down 1
    x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name="down1_conv1")(inputs)
    x = BatchNormalization(name="down1_bn1")(x)
    x = Activation("relu", name="down1_relu1")(x)
    x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name="down1_conv2")(x)
    x = BatchNormalization(name="down1_bn2")(x)
    down1 = Activation("relu", name="down1_relu2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="down1_pool")(down1)

    # Down 2
    x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name="down2_conv1")(x)
    x = BatchNormalization(name="down2_bn1")(x)
    x = Activation("relu", name="down2_relu1")(x)
    x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name="down2_conv2")(x)
    x = BatchNormalization(name="down2_bn2")(x)
    down2 = Activation("relu", name="down2_relu2")(x)
    x = Dropout(0.5)(down2)
    x = MaxPooling2D(pool_size=(2, 2), name="down2_pool")(x)

    # Center
    x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name="center_conv1")(x)
    x = BatchNormalization(name="center_bn1")(x)
    x = Activation("relu", name="center_relu1")(x)
    x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name="center_conv2")(x)
    x = BatchNormalization(name="center_bn2")(x)
    x = Activation("relu", name="center_relu2")(x)
    x = Dropout(0.5)(x)

    # Up 1
    up1 = UpSampling2D(size=(2, 2), name="up1_upsampling")(x)
    x = concatenate([down2, up1], axis=3)
    x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name="up1_conv1")(x)
    x = BatchNormalization(name="up1_bn1")(x)
    x = Activation("relu", name="up1_relu1")(x)
    x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name="up1_conv2")(x)
    x = BatchNormalization(name="up1_bn2")(x)
    x = Activation("relu", name="up1_relu2")(x)

    # Up 2
    up2 = UpSampling2D(size=(2, 2), name="up2_upsampling")(x)
    x = concatenate([down1, up2], axis=3)
    x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name="up2_conv1")(x)
    x = BatchNormalization(name="up2_bn1")(x)
    x = Activation("relu", name="up2_relu1")(x)
    x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name="up2_conv2")(x)
    x = BatchNormalization(name="up2_bn2")(x)
    x = Activation("relu", name="up2_relu2")(x)

    # Output
    x = Conv2D(5, 3, padding='same', kernel_initializer='he_normal', name="output_conv")(x)
    x = BatchNormalization(name="output_bn")(x)
    outputs = Activation("sigmoid", name="output_activation")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def loss(y_true, y_pred):
    return BinaryCrossentropy()(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


def load_model(input_shape, weights=None):
    model = _unet(input_shape)
    model.compile(
        "Adam",
        loss=loss,
        metrics=[iou_coef, dice_coef],
    )

    if weights is not None:
        model.load_weights(weights)

    return model


def train_model(model):
    x, y = load_data(subset="train")

    checkpointer = ModelCheckpoint(filepath="vegdec_weights.h5", verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor="val_loss", patience=5)

    model.fit(
        x=x,
        y=y,
        batch_size=1,
        epochs=80,
        validation_split=0.2,
        callbacks=[checkpointer, earlystopping]
    )


def test_model(model):
    x, y = load_data(subset="test")
    return model.evaluate(x, y, batch_size=1)


def query_model(model, dataset="dataset1"):
    x, _ = load_data(subset=dataset, should_resize=True)
    y_pred = model.predict(x, batch_size=1)
    y_pred = process_output(y_pred, width=y_pred.shape[2], height=y_pred.shape[1])

    for i in range(len(x)):
        imsave(f"./data/outputs/{i+1}_rgb.png", np.uint8(np.round(x[i] * 255)))
        imsave(f"./data/outputs/{i+1}_mask.png", y_pred[i])

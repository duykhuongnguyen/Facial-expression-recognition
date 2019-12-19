import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
from tensorflow.keras import layers
from tensorflow.keras import models

def VGG19(num_classes, input_shape=(48, 48, 3), dropout=None, block5=True, batch_norm=True):
    img_input = layers.Input(shape=input_shape)

    #Block1
    x = layers.Conv2D(64, (3,3),
                      padding='same', 
                      name='block1_conv1')(img_input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3,3), 
                      padding='same', 
                      name='block1_conv2')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #Block2
    x = layers.Conv2D(128, (3,3),  
                      padding='same', 
                      name='block2_conv1')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, (3,3),  
                      padding='same', 
                      name='block2_conv2')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #Block3
    x = layers.Conv2D(256, (3,3), 
                      padding='same', 
                      name='block3_conv1')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(256, (3,3), 
                      padding='same', 
                      name='block3_conv2')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(256, (3,3), 
                      padding='same', 
                      name='block3_conv3')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(256, (3,3), 
                      padding='same', 
                      name='block3_conv4')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    #Block4
    x = layers.Conv2D(512, (3,3),  
                      padding='same', 
                      name='block4_conv1')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(512, (3,3),  
                      padding='same', 
                      name='block4_conv2')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(512, (3,3), 
                      activation='relu', 
                      padding='same', 
                      name='block4_conv3')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(512, (3,3),
                      padding='same', 
                      name='block4_conv4')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    #Block5
    if block5:
        x = layers.Conv2D(512, (3,3),  
                      padding='same', 
                      name='block5_conv1')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(512, (3,3),  
                        padding='same', 
                        name='block5_conv2')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(512, (3,3), 
                        activation='relu', 
                        padding='same', 
                        name='block5_conv3')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(512, (3,3),
                        padding='same', 
                        name='block5_conv4')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = layers.AveragePooling2D((1, 1), strides=(1, 1), name='block6_pool')(x)
    x = layers.Flatten()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    model = models.Model(img_input, x, name='vgg19')
    return model


if __name__ == '__main__':
    model = VGG19(input_shape=(48, 48 ,3), num_classes=7)
    print(model.summary())
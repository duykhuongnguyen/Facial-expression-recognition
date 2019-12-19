import tensorflow as tf
import keras
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from args import get_train_args, get_setup_args
from setup import preprocess
from sklearn.utils import class_weight
import numpy as np
from vgg import VGG19
from resnet import build_resnet

# Load data from file or preprocess
def data_loader(args):
    if args.load_from_path:
        try:
            X_train, y_train, X_dev, y_dev, X_test, y_test = np.load(args.train_faces), np.load(args.train_labels), np.load(args.dev_faces), np.load(args.dev_labels), np.load(args.test_faces), np.load(args.test_labels)
            weight = np.load('./data/weight.npy')
            print("Load data from file")
            return (X_train, y_train, X_dev, y_dev, X_test, y_test, weight)
        except:
            print("Can't load data from file. Preprocess data")
            return preprocess(args)

    else:
        return preprocess(args)

def train_generator():
    data_generator = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )
    return data_generator

def optimizer(args):
    sgd = optimizers.SGD(lr=args.lr, momentum=args.momentum, decay=args.lr/args.epochs)
    return sgd

def class_weights(labels):
    class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(labels),
                                                  labels)
    return class_weights

def callback(args):
    path = 'model/'+args.model+'_best.h5'
    checkpoint = ModelCheckpoint(path, verbose=1, monitor='val_acc',save_best_only=True, mode='auto')  
    return [checkpoint]

def build_model(args):
    if args.model == "VGG19":
        model = VGG19(num_classes=7, input_shape=(48, 48, 3), dropout=0.5)

    else:
        model = build_resnet(args.model, input_shape=(48, 48, 3), classes=7)
    return model

if __name__ == "__main__":
    args_ = get_train_args()
    # Load data
    X_train, y_train, X_dev, y_dev, X_test, y_test, weight = data_loader(args_)

    # Optimizer, image augmentation and add class weight
    sgd = optimizer(args_)
    train_generator = train_generator()
    class_weights = class_weights(weight)
    checkpoint = callback(args_)
    model = build_model(args_)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generator.flow(X_train, y_train, args_.batch_size),
                    steps_per_epoch = len(X_train)/args_.batch_size,
                    epochs=args_.epochs,
                    verbose=1,
                    validation_data=(X_dev, y_dev),
                    class_weight=class_weights,
                    callbacks=checkpoint)


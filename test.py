from tensorflow.keras.models import load_model
from args import get_train_args, get_setup_args
from setup import preprocess
from train_model import build_model, data_loader, optimizer

def load(path, args, compile=False):
    model = build_model(args)
    model.load_weights(path)
    sgd = optimizer(args)

    if compile:
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    args_ = get_train_args()

    model = load('model/ResNet18_best.h5', args_)


    X_train, y_train, X_dev, y_dev, X_test, y_test, weight = data_loader(args_)
    result = model.predict(X_test.argmax())
    print(result)
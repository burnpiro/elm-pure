import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from model import ELM


def main():
    num_classes = 10
    num_hidden_layers = 1024
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Process images into input vectors
    # each mnist image is a 28x28 picture with value ranges between 0 and 255
    x_train = x_train.astype(np.float32) / 255.
    x_train = x_train.reshape(-1, 28 ** 2)
    x_test = x_test.astype(np.float32) / 255.
    x_test = x_test.reshape(-1, 28 ** 2)

    # converts [1,2] into [[0,1,0], [0,0,1]]
    y_train = to_categorical(y_train, num_classes).astype(np.float32)
    y_test = to_categorical(y_test, num_classes).astype(np.float32)

    # create instance of our model
    model = ELM(
        28 ** 2,
        num_hidden_layers,
        num_classes
    )

    # Train
    model.fit(x_train, y_train)
    train_loss, train_acc = model.evaluate(x_train, y_train)
    print('train loss: %f' % train_loss)
    print('train acc: %f' % train_acc)

    # Validation
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print('val loss: %f' % val_loss)
    print('val acc: %f' % val_acc)


if __name__ == '__main__':
    main()

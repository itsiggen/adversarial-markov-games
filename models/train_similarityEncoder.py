from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, Activation, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

num_classes = 10
epochs = 10

#https://keras.io/examples/mnist_siamese/
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.maximum(K.square(margin) - K.square(y_pred), 0)
    return y_true * square_pred + (1 - y_true) * margin_square

def create_pairs_with_noise(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1 = digit_indices[d][i]
            for _ in range(20):
                mu, sigma = 0, 0.3 # mean and standard deviation
                s = np.random.normal(mu, sigma, 28*28)
                s = s.reshape(28,28,1)
                s = np.absolute(s)
                z2 = np.add(np.asarray(x[z1]), s)
                
                pairs += [[x[z1], z2.tolist()]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
            
    return np.array(pairs).astype('float32'), np.array(labels).astype('float32')

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', name='conv2d_1',input_shape=input_shape))
    model.add(Activation('relu', name='activation_1'))
    model.add(Conv2D(32, (3, 3), name='conv2d_2'))
    model.add(Activation('relu', name='activation_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))
    model.add(Dropout(0.25, name='dropout_1'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv2d_3'))
    model.add(Activation('relu', name='activation_3'))
    model.add(Conv2D(64, (3, 3), name='conv2d_4'))
    model.add(Activation('relu', name='activation_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
    model.add(Dropout(0.25, name='dropout_2'))

    model.add(Flatten(name='flatten_1'))
    model.add(Dense(512, name='dense_1'))
    model.add(Activation('relu', name='activation_5'))
    model.add(Dropout(0.5, name='dropout_3'))
    model.add(Dense(256, name='dense_encode'))  #256 is the encoding dim in the stateful detection paper
    model.add(Activation('linear', name='encoding'))
    
    return model


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[0:10000]
y_train = y_train[0:10000]

x_train = x_train.astype('float32').reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.astype('float32').reshape(x_test.shape[0], 28, 28, 1)

x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

# create training/test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs_with_noise(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs_with_noise(x_test, digit_indices)

# network definition
base_network = create_base_network(input_shape)
print("passed")

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# re-use the same base_network to share the weights across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

base_network.save('MNISTencoder.h5')
base_network.save_weights('MNISTencoderweights.h5')

print(model.summary())
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
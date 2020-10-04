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

def create_pairs_met_noise(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1 = digit_indices[d][i]
            for _ in range(2):
                mu, sigma = 0, 0.2 # mean and standard deviation
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
            
    return np.array(pairs), np.array(labels)

def create_pairs_vergelijk_met_andere(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

def create_base_network_zelf(input_shape):
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
    model.add(Dense(256, name='dense_encode'))  #256 is de encode dim van de paper (zie code)
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
#model keras acc=0.9909
# results
# [(0, 0.016529655),
#   (1, 1.0318576),
#   (2, 0.010345819),
#   (3, 1.0195526),
#   (4, 0.0059748082),
#   (5, 1.291814),
#   (6, 0.012074598),
#   (7, 1.0146618),
#   (8, 0.019244136),
#   (9, 1.288227),
#   (10, 0.019910207),
#   (11, 1.0458918),
#   (12, 0.007295643),
#   (13, 1.0258605),
#   (14, 0.00882173),
#   (15, 1.2631898),
#   (16, 0.011830627),
#   (17, 0.87516505),
#   (18, 0.010770255),
#   (19, 1.2776662),
#   (20, 0.0038376367),
#   (21, 1.0277332),
#   (22, 0.010177514),
#   (23, 1.3369346),
#   (24, 0.015541091),
#   (25, 1.297),
#   (26, 0.016101668),
#   (27, 1.2549115),
#   (28, 0.014614203),
#   (29, 1.0230094),
#   (30, 0.000959303),
#   (31, 1.255126),
#   (32, 0.009096499),
#   (33, 0.9683829),
#   (34, 0.0040347204),
#   (35, 1.2825375),
#   (36, 0.00611698),
#   (37, 1.0192435),
#   (38, 0.0053327605),
#   (39, 0.99100184),
#   (40, 0.007525443),
#   (41, 1.0106492),
#   (42, 0.00096788775),
#   (43, 1.0358847),
#   (44, 0.010931428),
#   (45, 1.0268378),
#   (46, 0.017410891),
#   (47, 0.95844215),
#   (48, 0.008505884),
#   (49, 1.2943285),
#   (50, 0.021308793),
#   (51, 1.0109044),
#   (52, 0.005860963),
#   (53, 1.0075043),
#   (54, 0.00822908),
#   (55, 1.2987138),
#   (56, 0.018725755),
#   (57, 1.0387529),
#   (58, 0.023251306),
#   (59, 1.2901418),
#   (60, 0.01630395),
#   (61, 1.2977645),
#   (62, 0.008793995),
#   (63, 1.0082757),
#   (64, 0.003559334),
#   (65, 1.261476),
#   (66, 0.011365198),
#   (67, 1.0226774),
#   (68, 0.018556926),
#   (69, 1.3098984),
#   (70, 0.005299762),
#   (71, 1.3171041),
#   (72, 0.007023495),
#   (73, 0.9777178),
#   (74, 0.010717765),
#   (75, 0.96634287),
#   (76, 0.0033248768),
#   (77, 1.3000994),
#   (78, 0.01732437),
#   (79, 1.2837888),
#   (80, 0.0016819838),
#   (81, 1.2717298),
#   (82, 0.02720159),
#   (83, 1.2751089),
#   (84, 0.023401422),
#   (85, 1.030369),
#   (86, 0.010012156),
#   (87, 1.2744932),
#   (88, 0.0034034585),
#   (89, 1.2575214),
#   (90, 0.016545895),
#   (91, 1.2843059),
#   (92, 0.0104796225),
#   (93, 1.3118044),
#   (94, 0.013800181),
#   (95, 1.0155737),
#   (96, 0.011276747),
#   (97, 1.2994266),
#   (98, 0.013606909),
#   (99, 0.96280193)]
'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
'''
#model zelf acc=0.9913
# [(0, 0.009579939),
#  (1, 1.177692),
#  (2, 0.007892452),
#  (3, 1.0732976),
#  (4, 0.011612723),
#  (5, 1.0766203),
#  (6, 0.02433205),
#  (7, 0.92949927),
#  (8, 0.012683397),
#  (9, 0.9128857),
#  (10, 0.017789617),
#  (11, 1.0117927),
#  (12, 0.076560326),
#  (13, 1.0825132),
#  (14, 0.036925886),
#  (15, 0.9930283),
#  (16, 0.045480665),
#  (17, 1.016766),
#  (18, 0.017767984),
#  (19, 0.9588616),
#  (20, 0.021869795),
#  (21, 1.0145011),
#  (22, 0.00562875),
#  (23, 0.96744674),
#  (24, 0.013256374),
#  (25, 0.99400795),
#  (26, 0.026347613),
#  (27, 1.0454242),
#  (28, 0.039266326),
#  (29, 0.8311995),
#  (30, 0.021369865),
#  (31, 1.0597262),
#  (32, 0.025870677),
#  (33, 0.93998724),
#  (34, 0.016741654),
#  (35, 1.0682511),
#  (36, 0.013348725),
#  (37, 1.1010689),
#  (38, 0.050498564),
#  (39, 1.0449272),
#  (40, 0.05254212),
#  (41, 1.0758117),
#  (42, 0.009215093),
#  (43, 0.96403986),
#  (44, 0.012051951),
#  (45, 1.0541022),
#  (46, 0.04664682),
#  (47, 1.0338107),
#  (48, 0.0270396),
#  (49, 1.0789255),
#  (50, 0.050895996),
#  (51, 1.037478),
#  (52, 0.1582753),
#  (53, 0.9252905),
#  (54, 0.22875364),
#  (55, 1.00141),
#  (56, 0.048818),
#  (57, 1.0855078),
#  (58, 0.024451308),
#  (59, 0.9633015),
#  (60, 0.01733033),
#  (61, 1.1846611),
#  (62, 0.004561268),
#  (63, 1.1359057),
#  (64, 0.013458098),
#  (65, 1.0033296),
#  (66, 0.0072259554),
#  (67, 0.9093129),
#  (68, 0.025454283),
#  (69, 1.041318),
#  (70, 0.004997599),
#  (71, 1.0106342),
#  (72, 0.027800918),
#  (73, 0.9563257),
#  (74, 0.008178282),
#  (75, 1.0246946),
#  (76, 0.008017215),
#  (77, 1.039952),
#  (78, 0.018873371),
#  (79, 1.0208217),
#  (80, 0.016857876),
#  (81, 1.0434254),
#  (82, 0.042223766),
#  (83, 1.063612),
#  (84, 0.03798299),
#  (85, 1.0596781),
#  (86, 0.007915534),
#  (87, 0.94438624),
#  (88, 0.010653474),
#  (89, 1.0054647),
#  (90, 0.019840352),
#  (91, 1.0661063),
#  (92, 0.0076361974),
#  (93, 1.035416),
#  (94, 0.025269743),
#  (95, 0.9378113),
#  (96, 0.022945767),
#  (97, 1.0789827),
#  (98, 0.035287455),
#  (99, 0.9315769)]
x_train = x_train.astype('float32').reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.astype('float32').reshape(x_test.shape[0], 28, 28, 1)

x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs_met_noise(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs_met_noise(x_test, digit_indices)


#uitAnders2 = [(numpy.subtract(base_network.predict(tr_pairs[i][0].reshape(1,28,28)),base_network.predict(tr_pairs[i][1].reshape(1,28,28)))) for i in range(100)]
#uit = [(i,np.linalg.norm(x[0])) for i,x in enumerate(uitAnders2)]
# network definition
base_network = create_base_network_zelf(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
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

base_network.save('SIMILARITYmodel.h5')
base_network.save_weights('SIMILARITYmodelWeights.h5')

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
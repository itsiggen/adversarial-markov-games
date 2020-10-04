import tensorflow as tf
import eagerpy as ep
import numpy as np
from foolbox import TensorFlowModel, accuracy, samples
from foolbox.attacks import BoundaryAttack, HopSkipJump
from tensorflow.keras.models import Sequential, load_model
from foolbox.criteria import TargetedMisclassification


# instantiate a model
# model = tf.keras.applications.ResNet50(weights="imagenet")
# pre = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
# fmodel = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)

# instantiate a MNIST model
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

model = load_model('models/MNISTmodel.h5')
fmodel = TensorFlowModel(model, bounds=(0, 1))
criterion = TargetedMisclassification(tf.convert_to_tensor([7], dtype='float32'))

startImgAttack = x_test[0]
wantedImgAttack = x_test[1]

# attacker = BoundaryAttack()       
# attack = attacker.run(fmodel,wantedImgAttack.reshape(1, 28, 28, 1), criterion, starting_points=startImgAttack.reshape(1, 28, 28, 1))
attack = BoundaryAttack()
epsilons = [0.3] 
advs, _, success = attack(fmodel, wantedImgAttack.reshape(1, 28, 28, 1), criterion, epsilons = epsilons, starting_points=startImgAttack.reshape(1, 28, 28, 1))

# # get data and test the model
# # wrapping the tensors with ep.astensors is optional, but it allows
# # us to work with EagerPy tensors in the following
# images, labels = ep.astensors(*samples(fmodel, dataset="imagenet"))
# print(accuracy(fmodel, images, labels))

# # apply the attack
# attack = BoundaryAttack()
# epsilons = [0.3]
# advs, _, success = attack(fmodel, images, labels, epsilons=epsilons)

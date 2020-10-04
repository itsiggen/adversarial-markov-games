# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:34:37 2020

@author: rikp
"""
import tensorflow as tf
from buckets import Buckets

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],1, 28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

buckets = Buckets()
related = 0
#2388 related 10 buckets (related gezien als er 1 van de 10 buckets al related was)
#0 keer tot 30 geraakt
for benignExample in x_test:
    nr,state,r = buckets.getBucketAndStateForQuery(benignExample)
    buckets.addQuery_LabelPredicted_LabelReturned_RealLabel(benignExample,1,1,nr)
    if not (state is None):
        related += r
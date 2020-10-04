import numpy as np # linear algebra
from sklearn import metrics
from sklearn import tree
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import time
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


############################################
## DECISION TREE CLASSIFIER (acc: 87,79%) ##
############################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(x_train), len(y_train))
print(x_train[1].shape)
if False:
    clf1 = tree.DecisionTreeClassifier()
    if False:
        print("start training DTC")
        clf1 = clf1.fit(x_train, y_train)
        joblib.dump(clf1, 'DTC.pkl') 
    else:
        clf1 = joblib.load('../models/DTC.pkl')
        predicted = clf1.predict(x_test)
        print("Accuracy: ", accuracy_score(y_test, predicted))
        start = time.time()
        clf1.predict([x_test[1]])
        end = time.time()
        print("time needed ",end-start)
##################################################
## K NEAREST NEIGHBORS CLASSIFIER (acc: 0.9688) ##
##################################################
if False:
    clf2 = KNeighborsClassifier()
    if False:
        print("start training KNNC")
        clf2.fit(x_train, y_train)
        joblib.dump(clf2, 'KNNC.pkl') 
    else:
        clf2 = joblib.load('../models/KNNC.pkl')
        #predicted = clf2.predict(x_test)
        #print("Accuracy: ", accuracy_score(y_test, predicted))
        start = time.time()
        clf2.predict([x_test[1]])
        end = time.time()
        print("time needed ",end-start)
############################################
## RANDOM FOREST CLASSIFIER (acc: 0.9688) ##
############################################
if False:
    clf3 = RandomForestClassifier(n_estimators=100)
    if False:
        print("start training RFC")
        clf3.fit(x_train, y_train)
        joblib.dump(clf3, 'RFC.pkl')
    else:
        clf3 = joblib.load('../models/RFC.pkl')
        predicted = clf3.predict(x_test)
        print("Accuracy: ", accuracy_score(y_test, predicted))
        start = time.time()
        clf3.predict([x_test[1]])
        end = time.time()
        print("time needed ",end-start)

###########################################
## Support vector machine SVM acc 94,17% ##
## time needed 0.0209                    ##
## boundary attack failed 35/50 keer     ##
###########################################
if True:
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)
    
    classifier.fit(x_train, y_train)
    
    predicted = classifier.predict(x_test)
    
    print("Accuracy: ", accuracy_score(y_test, predicted))
    
    start = time.time()
    classifier.predict([x_test[1]])
    end = time.time()
    print("time needed ",end-start)
    
    joblib.dump(classifier, 'SVM.pkl')














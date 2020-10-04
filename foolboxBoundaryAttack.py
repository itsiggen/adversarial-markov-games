"""
import matplotlib
#matplotlib.use('Agg')
"""
import gc

import tensorflow as tf
import eagerpy as ep
from foolbox.models.base import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from foolbox.attacks import BoundaryAttack
import pickle
import collections
import random
import joblib
import time
from statistics import mean
import math
import sys
import math

from tensorflow.keras.utils import plot_model

from datetime import datetime

from replay_buffer import ReplayBuffer
from deep_Q import DeepQ
from buckets import Buckets

from foolbox.criteria import Misclassification,TargetedMisclassification
from foolbox.models.tensorflow import TensorFlowModel

'''
start = time.time()
print("hello")
end = time.time()
print(end - start)
'''

# /cw/thesis-r0622635/miniconda/envs/thesis/lib/python3.7/site-packages/foolbox
# /cw/thesis-r0622635/
# C:\Users\rikp\Anaconda3\envs\thesis\Lib\site-packages\foolbox
# C:\Users\rikp\Documents\thesisproject\rik-pauwels
#todo
# state representation met enkel history predicted labels, returned labels proberen?
RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)

RUNDEPARTEMENT = True

def showImg(extra = ""):
    if RUNDEPARTEMENT:
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H_%M_%S")
        plt.savefig('./plots/img'+date_time+extra+'.png')
    else:
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H_%M_%S")
        plt.savefig('./plots/img'+date_time+extra+'.png')

class TrainDefenderModel(Model):
    
    def __init__(self, targetModel, bounds, rewarder, modelSwapModel):
        self.targetModel = targetModel
        
        self.actionsPreformed = [0,0,0,0]
        self.bucketsUsed = [0 for i in range(10)]
        
        self.queryCounter = 0
        self.bounds = bounds#Bounds(*bounds)
        self.rewarder = rewarder
        self.modelSwapModel = modelSwapModel
        
        #RL parameters
        self.tau = 0.01
        self.decay_rate = 0.99
        self.buffer_size = 50000
        self.minibatch_size = 32
        self.tot_queries = 300000 #1000000 #300000
        self.epsilon_decay = 75000 #300000 # 75000 #dit is de a van de rechte y=ax+b die het verloop van de epsilon geeft
        self.final_epsilon = 0.1
        self.initial_epsilon = 1.0
        self.tau = 0.01
        
        self.startTrainingQuery = 200
        
        self.epsilon = self.initial_epsilon
        self.total_reward = 0
        
        #self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.replay_buffer_benign = ReplayBuffer(self.buffer_size)
        self.replay_buffer_adversarial = ReplayBuffer(self.buffer_size)
        
        self.deep_q = DeepQ((435 + 30,),3) #het is een 30 op 30 is dan is de flatlist (30*30/2)-30/2=435 + 30 van labeled + 30 van returned 
        
        self.countAttackQueries = 0
        
        self.firstRun = True
        
        self.startTime = time.time()
        self.beginTime = time.time()
        
        self.buckets = Buckets(nrBuckets=10)
        
        self.correctBenign = 0
        self.totalBenignClassified = 0
        
        
        
    def __call__(self, inputs, isOriginalAttackQuery=True, realLabelBeneignQuery=None):
        
        ################################################################
        ## UNIMPORTANT PARAMETER UPDATES EN LOGGING CODE FOR TRAINING ##
        ################################################################
        if isOriginalAttackQuery:
            self.queryCounter += 1
            self.countAttackQueries += 1
            if self.countAttackQueries%20000 == 0:
                print("run ", self.countAttackQueries, "took ", time.time()-self.startTime, " seconds")
                self.startTime = time.time()
        
        if self.firstRun:
            #print('actionsPreformed en count attack zijn gereset')
            self.actionsPreformed = [0,0,0,0]
            self.countAttackQueries = 0
            self.firstRun = False
            self.total_reward = 0
            self.buckets = Buckets()
            self.correctBenign = 0
            self.totalBenignClassified = 0
            self.startTime = time.time()
            self.beginTime = time.time()
            self.bucketsUsed = [0 for i in range(10)]
        
        ##############################
        ## SIMULATE BENEIGN QUERIES ##
        ##############################
        AANTAL_BENEIGN_QUERIES = 1
        if isOriginalAttackQuery:
            for i in range(AANTAL_BENEIGN_QUERIES):
            #1 pick query
                if False:
                    nr = random.randint(0,50000)
                    benignQuery = x_train[nr]
                else:
                    nr=self.queryCounter//100
                    
                    mu, sigma = 0, 0.1 # mean and standard deviation
                    s = np.random.normal(mu, sigma, 28*28)
                    s = s.reshape(28,28,1)
                    s = np.absolute(s)
                    benignQuery = np.add(x_train[nr], s)
                    
            #2 call label function
                beneignQuery = ep.astensor(benignQuery.reshape(1, 28, 28, 1))
                realLabelBeneignQuery=y_train[nr]
                self.__call__(beneignQuery,isOriginalAttackQuery=False,realLabelBeneignQuery=realLabelBeneignQuery)
        
        ###################
        ## PROCESS QUERY ##
        ###################
        x, restore_type = ep.astensor_(inputs)
        
        bucketIndex,curr_state = self.buckets.getBucketAndStateForQuery(x.raw)
        
        self.bucketsUsed[bucketIndex]+=1
        
        if self.buckets.getIfBucketIsFullBucket(bucketIndex):
            if isOriginalAttackQuery:
                reward = self.rewarder.calculate_reward_adversarial(self.buckets.getAverageStepSizeBucket(bucketIndex),self.countAttackQueries)
            else:
                reward = self.rewarder.calculate_reward_beneign(self.buckets.getLastReturnedLabelBucket(bucketIndex),self.buckets.getLastRealLabel(bucketIndex))
            sr = self.buckets.getLastStateRepresentationBucket(bucketIndex)
            
            if sr is not None:          
#                self.replay_buffer.add(sr, self.buckets.getLastActionBucket(bucketIndex), reward, curr_state)
                
                if isOriginalAttackQuery:
                    self.replay_buffer_adversarial.add(sr, self.buckets.getLastActionBucket(bucketIndex), reward, curr_state)
                else:
                    self.replay_buffer_benign.add(sr, self.buckets.getLastActionBucket(bucketIndex), reward, curr_state)
            
            self.total_reward += reward 
        
        ###########################################
        ## CHECK IF THERE IS A STATE TO ACT UPON ##
        ###########################################
        if curr_state is None:
            predictedAction = 2
            returnedPredictions = self.targetModel.predict(x.raw)
            self.actionsPreformed[3]+=1
        else:
            
            ####################
            ## PREDICT ACTION ##
            ####################
            if self.epsilon > self.final_epsilon:
                self.epsilon -= (self.initial_epsilon-self.final_epsilon)/self.epsilon_decay
            
            predictedAction = self.deep_q.predict_action(curr_state, self.epsilon)
            
            ####################
            ## PREFORM ACTION ##
            ####################
            self.actionsPreformed[predictedAction] += 1
                # 0 is return second highest
            if predictedAction == 0:
                predictionTargetModel = self.targetModel.predict(x.raw)
                s=np.argsort(predictionTargetModel)
                returnedPredictions = np.where(s==s.shape[1]-2,1,0)

                # 1 is model swapping
            if predictedAction == 1:
                xArray = np.asarray(x.raw)
                xArray = xArray.reshape(1,28*28)
                predictedClassSwap = self.modelSwapModel.predict(xArray)
                returnedPredictions = [0 for _ in range(10)]
                returnedPredictions[predictedClassSwap[0]]=1
                returnedPredictions = np.asarray([returnedPredictions])
                    
                # 2 is doe niets
            if predictedAction == 2:
                returnedPredictions = self.targetModel.predict(x.raw)
                
                # 3 is add input noise
            if False and predictedAction == 3:
                mu, sigma = 0, 1 # mean and standard deviation
                s = np.random.normal(mu, sigma, 28*28)
                s = s.reshape(1,28,28,1)
                s = np.absolute(s)
                x = np.add(x.raw, s)
                returnedPredictions = self.targetModel.predict(x)
            ####################################
            ## TRAIN REWARDER ON SEEN QUERIES ##
            ####################################
            
            if self.queryCounter > self.startTrainingQuery and self.countAttackQueries%200==0:
                #s_batch, a_batch, r_batch, s2_batch = self.replay_buffer.sample(100)
                # self.deep_q.train(s_batch, a_batch, r_batch, s2_batch, 0.8)
                
                s_batch, a_batch, r_batch, s2_batch = self.replay_buffer_adversarial.sample(100)
                self.deep_q.train(s_batch, a_batch, r_batch, s2_batch, 0.8)
                
                s_batch, a_batch, r_batch, s2_batch = self.replay_buffer_benign.sample(100)
                self.deep_q.train(s_batch, a_batch, r_batch, s2_batch, 0)
                
                self.deep_q.target_train()
        
        predictionTargettedModel = self.targetModel.predict(x.raw)

        self.buckets.addQuery_Action_LabelPredicted_LabelReturned_RealLabel(x.raw,predictedAction,np.argmax(predictionTargettedModel),np.argmax(returnedPredictions),bucketIndex,realLabelBeneignQuery)
        
        
        if not isOriginalAttackQuery:
            self.totalBenignClassified += 1
            if realLabelBeneignQuery == np.argmax(returnedPredictions):
                self.correctBenign += 1
        
        z = ep.astensor(returnedPredictions)    
        
        return restore_type(z)

    def bounds(self):
        return self._bounds
    
    def getQueryCount(self):
        return self.queryCounter 
    
    def storeModel(self,date_time = ''):
        message = 'models/ModelReinforcementlearner_'+date_time+'1relatedBenign300000_750000Buckets2replayBufferMeerdereBenignTrain_bucket0-1_reward0-5_andereState_geenQueryCounter_base10.h5'
        print(message)
        self.deep_q.save_network(message)
        self.deep_q.save_network('models/RLmodel2.h5')


# normalize
class Rewarder():
    def calculate_reward_adversarial(self,averageStepsize,queryCounter):
        
        if averageStepsize <= 0:
            print("te lage average stepsize: ",averageStepsize)
            averageStepsize = 0
        
        #reward based on averageStepsize
        if averageStepsize >= 1:
            r = 0#-queryCounter/1000
        else:
            #print(averageStepsize)
            r = abs(math.log(averageStepsize,10))#-queryCounter/1000
            
        return r
    
    def calculate_reward_beneign(self,returnedLabel=None,realLabel=None):
        if returnedLabel != realLabel:
            return -5
        else:
            return 0


class ExecuteDefenderModel(Model):
    
    def __init__(self, target_model, bounds, deep_q, swap_model):
        self.target_model = target_model
        self.deep_q = deep_q
        self.swap_model = swap_model
        self.bounds = bounds
        self.countAttackQueries = 0

        self.time = time.time()
        
        self.buckets = Buckets()
        
        self.actionsPreformed = [0,0,0,0]
        
    def bounds(self):
        return self.bounds
    
    def clear(self):
        self.countAttackQueries = 0
        self.actionsPreformed = [0,0,0,0]
    
    def __call__(self,inputs):
        self.countAttackQueries += 1
        x, restore_type = ep.astensor_(inputs)
        
        if self.countAttackQueries%20000 == 0:
            print("run ", self.countAttackQueries)
            end = time.time()
            print("het duurde ", end - self.time, " seconds")
            self.time = end
        
        bucketIndex,history = self.buckets.getBucketAndStateForQuery(x.raw)
        
        curr_state = history
        
        predictionTargettedModel = target_model.predict(x.raw)
    
        
        if curr_state is None:
            predictedAction = 2
            returnedPredictions = predictionTargettedModel
            self.actionsPreformed[3]+=1
        else:                      
            ####################
            ## PREFORM ACTION ##
            ####################
                # 0 is return second highest
            
            predictedAction=self.deep_q.predict_action(curr_state,0)
            self.actionsPreformed[predictedAction]+=1
            if predictedAction == 0:
                predictionTargetModel = predictionTargettedModel.copy()
                s=np.argsort(predictionTargetModel)
                returnedPredictions = np.where(s==s.shape[1]-2,1,0)

                # 1 is model swapping
            if predictedAction == 1:
                xArray = np.asarray(x.raw)
                xArray = xArray.reshape(1,28*28)
                predictedClassSwap = self.swap_model.predict(xArray)
                returnedPredictions = [0 for _ in range(10)]
                returnedPredictions[predictedClassSwap[0]]=1
                returnedPredictions = np.asarray([returnedPredictions])
                    
                # 2 is doe niets
            if predictedAction == 2:
                returnedPredictions = predictionTargettedModel
                
                # 3 is add input noise
            if False and predictedAction == 3:
                mu, sigma = 0, 1 # mean and standard deviation
                s = np.random.normal(mu, sigma, 28*28)
                s = s.reshape(1,28,28,1)
                s = np.absolute(s)
                x = np.add(x.raw, s)
                returnedPredictions = self.targetModel.predict(x)
        
        self.buckets.addQuery_LabelPredicted_LabelReturned_RealLabel(x.raw,np.argmax(predictionTargettedModel),np.argmax(returnedPredictions),bucketIndex)
        
        z = ep.astensor(returnedPredictions)
        return restore_type(z)
    
    
    
class ExecuteRandomDefenderModel(Model):
    
    def __init__(self, target_model, bounds, swap_model):
        self.target_model = target_model
        self.swap_model = swap_model
        self.bounds = bounds
        self.countAttackQueries = 0
        
        self.time = time.time()
        
        self.actionsPreformed = [0,0,0,0]
        
    def bounds(self):
        return self.bounds
    
    def clear(self):
        self.countAttackQueries = 0
        self.actionsPreformed = [0,0,0,0]
    
    def __call__(self,inputs):
        self.countAttackQueries += 1
        x, restore_type = ep.astensor_(inputs)
        
        if self.countAttackQueries%20000 == 0:
            print("run ", self.countAttackQueries)
            end = time.time()
            print("het duurde ", end - self.time, " seconds")
            self.time = end
            
        predictedAction = np.random.randint(0,3)
        
        predictionTargettedModel = target_model.predict(x.raw)
        
        self.actionsPreformed[predictedAction]+=1
        if predictedAction == 0:
            predictionTargetModel = predictionTargettedModel.copy()
            s=np.argsort(predictionTargetModel)
            returnedPredictions = np.where(s==s.shape[1]-2,1,0)

            # 1 is model swapping
        if predictedAction == 1:
            xArray = np.asarray(x.raw)
            xArray = xArray.reshape(1,28*28)
            predictedClassSwap = self.swap_model.predict(xArray)
            returnedPredictions = [0 for _ in range(10)]
            returnedPredictions[predictedClassSwap[0]]=1
            returnedPredictions = np.asarray([returnedPredictions])
                
            # 2 is doe niets
        if predictedAction == 2:
            returnedPredictions = predictionTargettedModel
            
            # 3 is add input noise
        if False and predictedAction == 3:
            mu, sigma = 0, 1 # mean and standard deviation
            s = np.random.normal(mu, sigma, 28*28)
            s = s.reshape(1,28,28,1)
            s = np.absolute(s)
            x = np.add(x.raw, s)
            returnedPredictions = self.targetModel.predict(x)
                
        self.actionsPreformed[predictedAction] += 1
        
        z = ep.astensor(returnedPredictions)
        return restore_type(z)
        
        
############################
## LOAD TARGET CLASSIFIER ##
############################
#https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
        
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

target_model = load_model('models/TargetMNISTmodel.h5')
target_model_logits = load_model('models/TargetMNISTmodelLogits.h5')

#######################
## LOAD SWAP MODEL(S) #
#######################

#RANDOM FOREST CLASSIFIER
swap_model = joblib.load('models/SVM.pkl') #RFC DTC


###########################
## LOAD SIMILARITY MODEL ##
###########################
similarity_model = load_model('models/SIMILARITYmodel.h5')


#######################
## INITIALISE ATTACK ##
#######################

rewarderObj = Rewarder()

# ziet een 4 maar was een 0, er staat een 0 op het scherm en computer ziet het als een 4
#origineel adversarial example
startImgNr = 0
startImg = x_train[startImgNr]  # nr 0 is een 5 

#image om naartoe te werken, dit label wil je hebben
goToNr = 1
goToImg = x_train[goToNr]
goToImgLabel = y_train[goToNr] # nr 1 is een 0 

attacker = BoundaryAttack(steps=50000) #na #steps+#steps*0,1+1 queries zal het stoppen


#######################
## MAKE OWN CRITERIA ##
#######################
class TargetedMisclassificationOriginalModel(TargetedMisclassification):
    
    def __init__(self,target_classes,target_model):
        super().__init__(target_classes)
        self.target_model = target_model
        self.counter = 0
        print("target class is", target_classes)
    
    #################################################
    ## NEEDED FOR CORRECT WORIKING BOUNDARY ATTACK ##
    #################################################
    def __call__(self,perturbed,outputs):
        if self.counter < 5:
            x, restore_type = ep.astensor_(perturbed)
            outputs = self.target_model.predict(x.raw)
            self.counter += 1
        else:
            x, restore_type = ep.astensor_(outputs)
            outputs = x.raw
            
        classes = outputs.argmax(axis=-1)
        assert classes.shape == self.target_classes.shape
        is_adv = classes == self.target_classes
        t = restore_type(is_adv)
        return t
    
#deze werken
#criterion = Misclassification(np.asarray([y_test[startImgNr]], dtype='float32'))
#criterion = TargetedMisclassification(np.asarray([goToImgLabel], dtype='float32'))
criterion = TargetedMisclassificationOriginalModel(np.asarray([goToImgLabel], dtype='float32'),target_model)

####################
## TRAIN DEFENDER ##
####################
train_defender = True

if train_defender:
    foolboxModel = TrainDefenderModel(target_model, (0,1), rewarderObj, swap_model)
     
    while True:
        #origineel adversarial example
        startImgNr = random.randint(0,1000)
        
        goToNr = random.randint(0,1000)

        while not np.argmax(target_model.predict(x_train[goToNr].reshape(1,28,28,1))) == y_train[goToNr]:
            goToNr = random.randint(0,1000)
        
        while y_train[startImgNr] == y_train[goToNr] or not np.argmax(target_model.predict(x_train[startImgNr].reshape(1,28,28,1))) == y_train[startImgNr]:
            startImgNr = random.randint(0,1000)
        
        startImg = x_train[startImgNr]  # nr 0 is een 5
        
        goToImg = x_train[goToNr]
        goToImgLabel = y_train[goToNr] # nr 1 is een 0
        
        criterion = TargetedMisclassificationOriginalModel(np.asarray([goToImgLabel], dtype='float32'),target_model)
        
        print("start img label ", y_train[startImgNr])
        
        attack = attacker.run(foolboxModel,startImg.reshape(1, 28, 28, 1), criterion,starting_points=goToImg.reshape(1, 28, 28, 1))
        
        if foolboxModel.queryCounter > foolboxModel.tot_queries:
            now = datetime.now()
            date_time = now.strftime("%m-%d-%Y-%H_%M_%S")
            foolboxModel.storeModel(date_time)
            break
        else:
            """
            try:
                """
            foolboxModel.firstRun = True
            criterion.firstRun = True
            print("random seed: ",RANDOM_SEED)
            print("countAttackQueries: ", foolboxModel.countAttackQueries)
            print(foolboxModel.actionsPreformed)
            print("predicted class: ", target_model.predict(attack.reshape(1,28,28,1)), "het wilt ", goToImgLabel," hebben")
            attackComp = attack.reshape((28,28))
            startImgComp = startImg.reshape((28,28))
            afstand = np.linalg.norm(attackComp-startImgComp)
            print("total reward: ", foolboxModel.total_reward)
            print("distance AE naar targetImg: ", afstand)
            print("queries done: ", foolboxModel.queryCounter)
            print("needed time: ",time.time()-foolboxModel.beginTime)
            plt.title("start img")
            plt.imshow(startImg.reshape((28, 28)), cmap='gray_r')
            showImg("first")
            plt.title("result img")
            plt.imshow(attack.reshape((28, 28)), cmap='gray_r')
            showImg()
            
            print("benign labeled ", foolboxModel.totalBenignClassified)
            print("benign correct labeled ", foolboxModel.correctBenign)
            res = foolboxModel.correctBenign/foolboxModel.totalBenignClassified if foolboxModel.correctBenign != 0 else 0
            print("procent benign juist: ",res)
            print("buckets used",foolboxModel.bucketsUsed)
            gc.collect()
            """
        except ValueError:
            print("FAILED (print van mezelf)")
            continue
        """
            
        print('begint opnieuw')

    print("countAttackQueries: ", foolboxModel.countAttackQueries)

############################
## TEST ON RELATED BENIGN ##
############################
testOnRelatedBenign = True
if testOnRelatedBenign:
    acc=0
    defender=DeepQ((435 + 30,),3)
    defender.load_network("models/RLmodel2.h5")
    target_model_foolbox = ExecuteDefenderModel(target_model,(0,1),defender,swap_model)
    
    #target_model_foolbox = ExecuteRandomDefenderModel(target_model, (0,1), swap_model)


    for i,benignExample in enumerate(x_test[:1000]):
        for j in range(100):
    
            mu, sigma = 0, 0.1 # mean and standard deviation
            s = np.random.normal(mu, sigma, 28*28)
            s = s.reshape(28,28,1)
            s = np.absolute(s)
            benignQuery = np.add(benignExample, s)
            
            lbl = np.argmax(target_model.predict(benignQuery.reshape(1,28,28,1)))
            predict = np.argmax(target_model_foolbox(benignQuery.reshape(1,28,28,1)))
            if lbl == predict:
                acc += 1
                
        print(target_model_foolbox.actionsPreformed)
    print("Accuracy is ", acc/100000)


##########################
## RUN WITHOUT DEFENDER ##
##########################
test_50_boundaryAttacks = True
#42 gelukt, 8 mislukt, mislukt (2,4,30,39,40,41,43,45,47) ste aanval
# 3 gelukt 47 fail

make_new_50 = False

if test_50_boundaryAttacks:
    if make_new_50:
        startImgAttackNrs = []
        wantedImgAttackNrs = []
        
        for i in range(50):
            startImgAttackNr = random.randint(0,1000)
            
            wantedImgAttackNr = random.randint(0,1000)
                
            while not np.argmax(target_model.predict(x_test[wantedImgAttackNr].reshape(1,28,28,1))) == y_test[wantedImgAttackNr]:
                wantedImgAttackNr = random.randint(0,1000)
            
            while y_test[startImgAttackNr] == y_test[wantedImgAttackNr] or not np.argmax(target_model.predict(x_test[startImgAttackNr].reshape(1,28,28,1))) == y_test[startImgAttackNr]:
                startImgAttackNr = random.randint(0,1000)
                
            startImgAttackNrs.append(startImgAttackNr)
            wantedImgAttackNrs.append(wantedImgAttackNr)
            
            filehandler = open('startImgAttackNrs.pickle', 'wb') 
            pickle.dump(startImgAttackNrs, filehandler)
            filehandler.close()
            
            filehandler = open('wantedImgAttackNrs.pickle', 'wb') 
            pickle.dump(wantedImgAttackNrs, filehandler)
            filehandler.close()
    
    else:
        filehandler = open('startImgAttackNrs.pickle', 'rb') 
        startImgAttackNrs = pickle.load(filehandler)
        filehandler.close()
        
        filehandler = open('wantedImgAttackNrs.pickle', 'rb') 
        wantedImgAttackNrs = pickle.load(filehandler)
        filehandler.close()
            
        
    #gemaakt omdat ik problemen had met eager mode       
    class TensorFlowModelOwn(Model):
        
        def __init__(self,model,bounds):
            self.model = model
            self.bounds = bounds
            self.countAttackQueries = 0
            
        def bounds(self):
            return self.bounds
        
        def __call__(self,inputs):
            self.countAttackQueries +=1
            x, restore_type = ep.astensor_(inputs)

            # z = self.model.predict(x.raw)
            # z=ep.astensor(z)
            
            
            xArray = np.asarray(x.raw)
            xArray = xArray.reshape(1,28*28)
            predictedClassSwap = self.model.predict(xArray)
            returnedPredictions = [0 for _ in range(10)]
            returnedPredictions[predictedClassSwap[0]]=1
            returnedPredictions = np.asarray([returnedPredictions])
            z=ep.astensor(returnedPredictions)
            
            t = restore_type(z)
            return t
    
    attacker = BoundaryAttack(steps=75000)
    
    failed = 0
    succeeded = 0 
    for i in range(len(startImgAttackNrs)):
        print("attack",i)
        print("BEZIG: ", succeeded, " GELUKT en ", failed, "MISLUKT")
        
        #test = y_test[startImgAttackNrs[i]].copy()
        
        h = x_test[startImgAttackNrs[i]].copy()
        
        
        # xArray = h.reshape(1,28*28)
        # lbl = swap_model.predict(xArray)[0] 
        
        lbl = np.argmax(target_model.predict(h.reshape(1,28,28,1)))
        
        criterion = TargetedMisclassificationOriginalModel(np.asarray([lbl], dtype='float32'),target_model)
        
        # target_model_foolbox = TensorFlowModelOwn(swap_model,(0,1))
        
        defender=DeepQ((435 + 30,),3)
        defender.load_network("models/RLmodel2.h5")
        target_model_foolbox = ExecuteDefenderModel(target_model,(0,1),defender,swap_model)
        
        #target_model_foolbox = ExecuteRandomDefenderModel(target_model, (0,1), swap_model)
        
        startImgAttack = x_test[startImgAttackNrs[i]]
        wantedImgAttack = x_test[wantedImgAttackNrs[i]]
        
        attack = attacker.run(target_model_foolbox,wantedImgAttack.reshape(1, 28, 28, 1), criterion,starting_points=startImgAttack.reshape(1, 28, 28, 1))
        
        # print("queryCount: ", target_model_foolbox.queryCounter)
        print("queryCount: ", target_model_foolbox.countAttackQueries)
        
        attackComp = attack.reshape((28,28))
        wantedImgAttackComp = wantedImgAttack.reshape((28,28))
        afstand = np.amax(abs(attackComp-wantedImgAttackComp))
        #afstand = np.linalg.norm(attackComp-wantedImgAttackComp, ord=np.inf)
        
        print("afstand: ",  afstand)
        
        # xArray = attack.reshape(1,28*28)
        # lbl = swap_model.predict(xArray)[0]
        
        lbl = np.argmax(target_model.predict(attack.reshape(1,28,28,1)))
        
        print("wanted label ", y_test[startImgAttackNrs[i]], " predicted  label ", lbl)
        
        if afstand < 0.3 and y_test[startImgAttackNrs[i]] == lbl:
            print("success ", i)
            succeeded += 1
        else:
            print("fail ", i)
            failed += 1
        
        
        plt.imshow(startImgAttack.reshape((28, 28)), cmap='gray_r')
        plt.title("start point")
        plt.show()

        plt.imshow(wantedImgAttack.reshape((28, 28)), cmap='gray_r')
        plt.title("wanted result")
        plt.show()
        
        plt.imshow(attack.reshape((28, 28)), cmap='gray_r')
        plt.title("adversarial example")
        plt.show()
        
        print("actions: ",target_model_foolbox.actionsPreformed)
                
    print("TOTAAL: ", succeeded, " GELUKT en ", failed, "MISLUKT")

######################################
## TEST ON UNRELATED BENIGN QUERIES ##
######################################
# 98.53% accuracy zonder dubbele replay buffer
testOnBenign = True
if testOnBenign:
    
    acc=0
    defender=DeepQ((435 + 30,),3)
    defender.load_network("models/RLmodel2.h5")
    target_model_foolbox = ExecuteDefenderModel(target_model,(0,1),defender,swap_model)
    
    #target_model_foolbox = ExecuteRandomDefenderModel(target_model, (0,1), swap_model)
    
    for i,benignExample in enumerate(x_test):
        lbl = np.argmax(target_model_foolbox(benignExample.reshape(1,28,28,1)))
        if lbl == y_test[i]:
            acc += 1
    
    print("Accuracy is ", acc/10000)

print('model found deze nacht 1 replay buffer')



#####################################
## RUN WITH RANDOM POLICY DEFENDER ##
#####################################
run_with_random_defender = False

if run_with_random_defender:
    foolboxModel = TrainDefenderModel(target_model,target_model_logits,(0,1),rewarderObj,swap_model,random_policy=True)
    attack = attacker.run(foolboxModel,startImg.reshape(1, 28, 28, 1), criterion,starting_points=goToImg.reshape(1, 28, 28, 1))
    print("queryCount: ", foolboxModel.queryCounter)


######################
## EXECUTE DEFENDER ##
######################
execute_defender = False

if execute_defender:
        
    defender=DeepQ((435 + 30 + 30,),3)
    defender.load_network("models/RLmodel_1bucket_pureBenign.h5")    
    
    filehandler = open('startImgAttackNrs.pickle', 'rb') 
    startImgAttackNrs = pickle.load(filehandler)
    filehandler.close()
    
    filehandler = open('wantedImgAttackNrs.pickle', 'rb') 
    wantedImgAttackNrs = pickle.load(filehandler)
    filehandler.close()
    
    failed = 0
    succeeded = 0 
    for i in range(len(startImgAttackNrs)):
        print("attack",i)
        print("BEZIG: ", succeeded, " GELUKT en ", failed, "MISLUKT")
        foolboxModel = ExecuteDefenderModel(target_model,target_model_logits,(0,1),defender,swap_model,similarity_model)
        
        startImgNr = startImgAttackNrs[i]
            
        wantedImgNr = wantedImgAttackNrs[i]
            
        startImg = x_test[startImgNr]  # nr 3 is een 0
        startImgLabel = y_test[startImgNr]
        wantedImg = x_test[wantedImgNr]
        wantedImgLabel = y_test[wantedImgNr] # nr 6 is een 4
        
        plt.imshow(startImg.reshape((28, 28)), cmap='gray_r')
        plt.title("start img")
        plt.show()
        plt.imshow(wantedImg.reshape((28, 28)), cmap='gray_r')
        plt.title("wanted img")
        plt.show()
        
        attackImg = startImg
        
        
        while foolboxModel.countAttackQueries < 1000:  

            criterion = TargetedMisclassificationOriginalModel(np.asarray([startImgLabel], dtype='float32'),target_model)
    
            attackImg = attacker.run(foolboxModel,wantedImg.reshape(1, 28, 28, 1), criterion,starting_points=attackImg.reshape(1, 28, 28, 1))
            print("queryCount: ", foolboxModel.countAttackQueries, "run: ", i)
            
            print("used actions: ", foolboxModel.actionsPreformed)
            
            attackComp = attackImg.reshape((28,28))
            startImgComp = startImg.reshape((28,28))
            wantedImgAttackComp = wantedImg.reshape((28,28))
            afstand = np.amax(abs(attackComp-wantedImgAttackComp))
            print("afstand=",afstand)
            
            print("wanted label=",startImgLabel)
            lbl = np.argmax(target_model.predict(attackImg.reshape(1,28,28,1)))
            print("prediction=",lbl)
            
        
        if afstand < 0.3 and y_test[startImgAttackNrs[i]] == lbl:
            print("success ", i)
            succeeded += 1
        else:
            print("fail ", i)
            failed += 1
        
        plt.imshow(attackImg.reshape((28, 28)), cmap='gray_r')
        plt.title("adversarial img")
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        """
        plotDataLogits = np.asarray(foolboxModel.loglogits)
        plotDataClassify = np.asarray(foolboxModel.logResult)     
        plotDataSimilarity = np.asarray(foolboxModel.logSimilarity)
    
        
        x = np.arange(plotDataClassify.shape[0])
        
        for i in range(128):
            plt.plot(x,plotDataLogits[:,0,i]) 
        
        plt.show()
    
        
        for i in range(10):
            if i == goToImgLabel or i == startImgLabel:
                plt.plot(x,plotDataClassify[:,0,i],label=i) 
            else:
                plt.plot(x,plotDataClassify[:,0,i]) 
        plt.legend(loc='lower right')
        plt.show()
        
        x = np.arange(plotDataSimilarity.shape[0])
        def L2Distance(x,y):
            r = 0
            for i in range(len(x)):
                r += (x[i]-y[i])**2
            return r**(1/2)
            
        yDistance = [L2Distance(plotDataSimilarity[0][0],e[0]) for e in plotDataSimilarity]
        
        plt.plot(x,yDistance)
        plt.show()
        """
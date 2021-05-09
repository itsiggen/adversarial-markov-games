import numpy as np
from tensorflow.keras.models import Sequential, load_model
from statistics import mean
import tensorflow as tf
import time

def l2(a,b):
    #L2 distance
    return np.linalg.norm(a-b)

similarityModel = load_model('models/SIMILARITYmodel.h5', compile=False)
def getSimilarityEncoding(query):
    # print(type(query))
    # npt = query.numpy()
    # tft = tf.convert_to_tensor(npt)
    s =  similarityModel.predict(query, steps=1)
    # z =  similarityModel.predict(tft, steps=1)
    # print(l2(s, z))
    return s

class Queues():
    def __init__(self, nrQueues=2, sizeState=30, threshold=0.1):
        self.queues = []        
        self.threshold = threshold
        self.sizeState = sizeState        
        self.nrQueues = nrQueues
        self.adds = 0
        self.lastQueueUsed = [i for i in range(nrQueues)]
        for i in range(nrQueues):
            self.queues.append(Queue(self.sizeState))
            
    # assign to queue based on minimum similarity encoding
    def addQuery(self, query, logits):
        simEnc = getSimilarityEncoding(query)
        # a = []
        # for i, queue in enumerate(self.queues):
        #     a.append(l2(queue.getLastEncoding(), simEnc))
        # # mb online update of threshold
        # if a[np.argmin(a)] < self.threshold:
        #     index = np.argmin(a)
        # else:
        #     index = 1
        if self.adds == 0:
            index = 0
        elif self.adds == 1:
            index = 1
        else:
            t = l2(self.queues[0].getLastEncoding(), simEnc)
            index = 0 if t < self.threshold else 1
        self.queues[index].addQueryToQueue(query, logits, simEnc)
        self.adds += 1
        # if self.adds > 2:
        #     print(l2(self.queues[index].encodings[-2], simEnc))
        return index

    def getState(self, index):
        return self.queues[index].getStateRepresentation()
               
    def getStepSizeQueue(self,queueNumber):
        # print(queueNumber)
        return self.queues[queueNumber].getStepSize()
    
    def getOriginQueue(self,queueNumber):
        return self.queues[queueNumber].getOrigin()
    
    def getLastStepQueue(self,queueNumber):
        return self.queues[queueNumber].getLastStep()
    
    def getQueueIsFull(self,queueNumber):
        return self.queues[queueNumber].getIfQueueIsFull()
    
    def getLastStateRepresentationQueue(self,queueNumber):
        return self.queues[queueNumber].getLastStateRepresentation()

class Queue():
    def __init__(self, sizeState):
        self.sizeState = sizeState
        self.sizeQueueMemory = 30
        self.amountOfQueries = 0
        self.encodings = []
        self.queryMemory = []
        self.stepsize = []
        self.logits = []
        self.starts = []
        self.origins = []

    def addQueryToQueue(self, query, logit, similaritySpaceEncoding):
        # self.lastBenignLabel = realLabel
        if self.amountOfQueries >= self.sizeQueueMemory:
            self.queryMemory.pop(0)
            self.encodings.pop(0)            
            self.stepsize.pop(0)
            self.logits.pop(0)
        if not self.stepsize:
            # average on MNIST
            self.stepsize.append(0.5)
        else:
            self.stepsize.append(l2(self.queryMemory[-1], query))
            
        self.encodings.append(similaritySpaceEncoding)
        self.logits.append(logit.squeeze().numpy())
        self.queryMemory.append(query)
        self.amountOfQueries += 1
        
        if self.amountOfQueries <= 49:
            rank = np.argsort(self.logits[-1])
            # print(rank)
            self.starts.append(rank[-1])
            # self.origins.append(rank[-2])
            self.setStartOrigin()
            # print(self.start)
            # print(self.origin)
        
        # if len(self.queryMemory) > 1:
        #     print(l2(self.queryMemory[-1], self.queryMemory[-2]))
        # print(self.stepsize[-1])
        
    def setStartOrigin(self):
        self.start = np.argmax(np.bincount(self.starts))
        # self.origin = np.argmax(np.bincount(self.origins))
        
    def getOrigin(self):
        rank = np.argsort(self.logits[-1])
        if rank[-1] == self.start:
            self.origin = rank[-2]
        else:
            self.origin = rank[-1]
        return self.origin
        
    def getLastEncoding(self):
        if len(self.encodings) == 0:
            return 0
        return self.encodings[-1]
    
    def getLastQuery(self):
        if len(self.queryMemory) == 0:
            return 0
        return self.queryMemory[-1]
    
    def getLastStep(self):
        return self.stepsize[-1]
    
    def getStepSize(self):
        return self.stepsize
    
    def getIfQueueIsFull(self):
        return len(self.queryMemory) >= self.sizeState
    
    def getStateRepresentation(self):
        # print(len(self.logits))
        arr = np.concatenate(self.logits)
        # print(arr.shape)
        state = np.zeros(300)
        state[0:len(arr)] = arr
        # print(state)
        return np.asarray(state)
    
    def getStateRepresentation2(self,timeInPast=0):
        lenQueryMem = len(self.queryMemory)
        low = max(lenQueryMem - self.sizeState - timeInPast, 0)
        high = max(lenQueryMem - timeInPast, 0)
        # Calculate all pairwise l2 distances
        # TODO: add cosine similarity as a proxy for perpendicular moves
        # consider simplified distances (last to previous 29)
        flat_list = [np.linalg.norm(self.queryMemory[i]-self.queryMemory[j]) \
                        for i in range(low, high) for j in range(low, i)]

        return np.asarray(flat_list)
        
    def getLastStateRepresentation(self):
        return self.getStateRepresentation(timeInPast=1) if len(self.queryMemory)-1 >= self.sizeState else None
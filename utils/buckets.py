import numpy as np
from tensorflow.keras.models import load_model
import timeit
import matplotlib.pyplot as plt

def l2(a,b):
    #L2 distance
    return np.linalg.norm(a-b)

similarityModel = load_model('models/SIMILARITYmodel.h5', compile=False)
def getSimilarityEncoding(query):
    s = similarityModel.predict(query)
    return s

class Buckets():
    def __init__(self, nrBuckets=10, sizeState=30, threshold=0.3):
        self.buckets = []
        self.enc = []
        self.threshold = threshold
        self.sizeState = sizeState
        self.nrBuckets = nrBuckets
        self.lastBucketsUsed = [i for i in range(nrBuckets)]
        
    def __createNewBucket__(self,indexOfBucketToReplace = None):
        if indexOfBucketToReplace == None:
            self.buckets.append(Bucket(self.sizeState))
        else:
            #print('indexOfBucketToReplace ',indexOfBucketToReplace)
            self.buckets[indexOfBucketToReplace] = Bucket(self.sizeState)
    
    def addQuery(self, query, action, logit, is_misdirection):
        # returns bucket and bucket.getStateRepresentation if within the similarity threshold
        # when no buckets within the threshold a new one is made, then added to the back of the list
        # returns the index of the bucket
        simEnc = getSimilarityEncoding(query)
        index = None
        for i,bucket in enumerate(self.buckets):
            a = l2(bucket.getLastEncoding(), simEnc)
            # print(a)
            if a < self.threshold:
                index = i
        if len(self.buckets)==self.nrBuckets and index is None:
            indexBucketToReplace = self.lastBucketsUsed[0]
            self.__createNewBucket__(indexBucketToReplace)
            index = indexBucketToReplace
        elif index is None:
            self.__createNewBucket__()
            index = len(self.buckets)-1
        self.buckets[index].addQueryToBucket(query, action, logit, simEnc, is_misdirection)
        self.lastBucketsUsed.remove(index)
        self.lastBucketsUsed.append(index)
        return index
                
    def getState(self, index):
        return self.buckets[index].getStateRepresentation()
               
    def getStepSizeBucket(self,bucketNumber):
        # print(bucketNumber)
        return self.buckets[bucketNumber].getStepSize()
    
    def getLastActionBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getLastAction()
    
    def getLastStepBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getLastStep()
    
    def getMisdirectionBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getMisdirection()
    
    def getIfBucketIsFullBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getIfBucketIsFull()
    
    def getLastRealLabel(self,bucketNumber):
        return self.buckets[bucketNumber].getLastRealLabel()
    
    def getLastStateRepresentationBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getLastStateRepresentation()

class Bucket():
    def __init__(self, sizeState):
        self.sizeState = sizeState
        self.sizeBucketMemory = 50
        self.amountOfQueries = 0
        self.encodings = []
        self.queryMemory = []
        self.stepsize = []
        self.misdirections = []
        self.actions = []
        self.logits = []
        self.starts = []
        self.origins = []
        self.lastRealLabel = None
    # queryMemory -> [query 1, query 2, query 3 ... query N]
    # after update
    # queryMemory -> [query 2, query 3, query 4 ... query N+1]
    def addQueryToBucket(self, query, action, logit, similaritySpaceEncoding, misdirection):
        # self.lastBenignLabel = realLabel
        if self.amountOfQueries >= self.sizeBucketMemory:
            self.queryMemory.pop(0)
            # self.averageSimilaritySpaceEncoding = ((self.sizeBucketMemory-1) * self.averageSimilaritySpaceEncoding + similaritySpaceEncoding)/(self.sizeBucketMemory)
            self.encodings.pop(0)            
            self.stepsize.pop(0)
            self.misdirections.pop(0)
            self.actions.pop(0)
            self.logits.pop(0)

        if not self.stepsize:
            # average on MNIST
            self.stepsize.append(0.11)
        else:
            # self.averageSimilaritySpaceEncoding = (self.amountOfQueries * self.averageSimilaritySpaceEncoding + similaritySpaceEncoding)/(self.amountOfQueries+1)
            self.stepsize.append(l2(self.queryMemory[-1], query))
            
        self.encodings.append(similaritySpaceEncoding)
        self.misdirections.append(misdirection)
        self.actions.append(action)
        self.logits.append(logit.squeeze().numpy())
        self.queryMemory.append(query)
        self.amountOfQueries += 1
        
        if self.amountOfQueries <= 49:
            rank = np.argsort(self.logits[self.amountOfQueries-1])
            # print(rank)
            self.starts.append(rank[-1])
            self.origins.append(rank[-2])
            self.setStartOrigin()
            # print(self.start)
            # print(self.origin)
        
        # if len(self.queryMemory) > 1:
        #     print(l2(self.queryMemory[-1], self.queryMemory[-2]))
        
    def setStartOrigin(self):
        self.start = np.argmax(np.bincount(self.starts))
        self.origin = np.argmax(np.bincount(self.origins))
        
    def getStartOrigin(self):
        return self.start, self.origin
        
    def getLastEncoding(self):
        return self.encodings[-1]
    
    def getLastAction(self):
        return self.actions[-1]
    
    def getLastStep(self):
        return self.stepsize[-1]

    def getMisdirection(self):
        return self.misdirections[-1]
    
    def getIfBucketIsFull(self):
        return len(self.queryMemory) >= self.sizeState
    
    def getLastRealLabel(self):
        return self.lastRealLabel
        
    def getStateRepresentation(self,timeInPast=0):
        lenQueryMem = len(self.queryMemory)
        low = max(lenQueryMem - self.sizeState - timeInPast, 0)
        high = max(lenQueryMem - timeInPast, 0)
        # Calculate all pairwise l2 distances
        # TODO: add cosine similarity as a proxy for perpendicular moves
        # consider simplified distances (last to previous 29)
        flat_list = [np.linalg.norm(self.queryMemory[i]-self.queryMemory[j]) \
                        for i in range(low, high) for j in range(low, i)]
        if lenQueryMem >= self.sizeState:
            misdirections = self.misdirections[-self.sizeState-timeInPast:len(self.misdirections)-timeInPast]
            flat_list.extend(misdirections)
        else:
            # Initialize query memory with zeroes
            flat_list.extend([0] * 465)
            flat_list = flat_list[:465]
    
        return np.asarray(flat_list)
        
    def getLastStateRepresentation(self):
        return self.getStateRepresentation(timeInPast=1) if len(self.queryMemory)-1 >= self.sizeState else None
        
    def getStepSize(self):
        # distances = np.asarray([np.linalg.norm(self.queryMemory[i]-self.queryMemory[i+1]) for i in range(len(self.queryMemory)-1)])
        # print(distances)
        # print(len(distances))
        # print(np.asarray(self.stepsize))
        # print(len(self.stepsize))
        return self.stepsize

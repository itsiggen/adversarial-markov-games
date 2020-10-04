import numpy as np
from tensorflow.keras.models import Sequential, load_model
from statistics import mean
import time


def getDistance(a,b):
    #L2 distance
    distance = np.linalg.norm(a-b)
    return distance

similarityModel = load_model('models/SIMILARITYmodel.h5')
def getSimilarityEncoding(query):
    s =  similarityModel.predict(query)
    return s

class Buckets():
    def __init__(self,nrBuckets=10,sizeState=30,threshold=0.1):
        self.buckets = []
        
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
    
    # returnt bucket en bucket.getStateRepresentation als binnen similarity threshold
    # als geen enkele binnen sim threshold -> nieuwe bucket aanmaken, als opgevraagd dan naar achter in lastbucketused
    def getBucketAndStateForQuery(self,query):
        simEnc = getSimilarityEncoding(query)
        for i,bucket in enumerate(self.buckets):
            if getDistance(bucket.getAverageSimilaritySpaceEncoding(),simEnc) < self.threshold:
                state = bucket.getStateRepresentation()
                return i,state
        if len(self.buckets)==self.nrBuckets:
            indexBucketToReplace = self.lastBucketsUsed[0]
                    
            self.__createNewBucket__(indexBucketToReplace)
            return indexBucketToReplace, self.buckets[indexBucketToReplace].getStateRepresentation()
        else:
            self.__createNewBucket__()
            index = len(self.buckets)-1
            return index,self.buckets[index].getStateRepresentation()
                
    def addQuery_Action_LabelPredicted_LabelReturned_RealLabel(self,query,action,predictedLabel,returnedLabel,bucketNumber,realLabel = None):
        similarityEncoding = getSimilarityEncoding(query)
        self.buckets[bucketNumber].addQueryToBucket(query,action,similarityEncoding,predictedLabel,returnedLabel,realLabel)
        self.lastBucketsUsed.remove(bucketNumber)
        self.lastBucketsUsed.append(bucketNumber)
        
    def addQuery_LabelPredicted_LabelReturned_RealLabel(self,query,predictedLabel,returnedLabel,bucketNumber):
        self.addQuery_Action_LabelPredicted_LabelReturned_RealLabel(query,None,predictedLabel,returnedLabel,bucketNumber)

        
    def getAverageStepSizeBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getAverageStepSize()
    
    def getLastActionBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getLastAction()
    
    def getLastPredictedLabelBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getLastPredictedLabel()
    
    def getLastReturnedLabelBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getLastReturnedLabel()
    
    def getIfBucketIsFullBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getIfBucketIsFull()
    
    def getLastRealLabel(self,bucketNumber):
        return self.buckets[bucketNumber].getLastRealLabel()
    
    def getLastStateRepresentationBucket(self,bucketNumber):
        return self.buckets[bucketNumber].getLastStateRepresentation()

class Bucket():
    def __init__(self,sizeState):
        self.sizeState = sizeState
        self.sizeBucketMemory = 50
        self.amountOfQueries = 0
        self.averageSimilaritySpaceEncoding = None
        self.queryMemory = []
        self.predictedLabels = []
        self.returnedLabels = []
        self.actions = []
        self.lastRealLabel = None
    # queryMemory -> [query 1, query 2, query 3 ... query 50]
    # na update
    # queryMemory -> [query 2, query 3, query 4 ... query 51]
    def addQueryToBucket(self,query,action,similaritySpaceEncoding,predictedLabel,returnedLabel,realLabel):
        self.lastRealLabel = realLabel
        if self.amountOfQueries >= self.sizeBucketMemory:
            self.queryMemory.pop(0)
            self.averageSimilaritySpaceEncoding = ((self.sizeBucketMemory-1) * self.averageSimilaritySpaceEncoding + similaritySpaceEncoding)/(self.sizeBucketMemory)
            
            self.predictedLabels.pop(0)
            self.predictedLabels.append(predictedLabel)
            
            self.returnedLabels.pop(0)
            self.returnedLabels.append(returnedLabel)
            
            self.actions.pop(0)
            self.actions.append(action)
        else:
            if self.averageSimilaritySpaceEncoding is None:
                self.averageSimilaritySpaceEncoding = similaritySpaceEncoding
                self.predictedLabels.append(predictedLabel)
                self.returnedLabels.append(returnedLabel)
                self.actions.append(action)
            else:
                self.averageSimilaritySpaceEncoding = (self.amountOfQueries * self.averageSimilaritySpaceEncoding + similaritySpaceEncoding)/(self.amountOfQueries+1)
                self.predictedLabels.append(predictedLabel)
                self.returnedLabels.append(returnedLabel)
                self.actions.append(action)

        self.queryMemory.append(query)
        self.amountOfQueries += 1
        
    def getAverageSimilaritySpaceEncoding(self):
        return self.averageSimilaritySpaceEncoding
    
    def getLastAction(self):
        return self.actions[-1]
    
    def getLastPredictedLabel(self):
        return self.predictedLabels[-1]
    
    def getLastReturnedLabel(self):
        return self.returnedLabels[-1]
    
    def getIfBucketIsFull(self):
        return len(self.queryMemory) >= self.sizeState
    
    def getLastRealLabel(self):
        return self.lastRealLabel
        
    def getStateRepresentation(self,timeInPast=0):
        if len(self.queryMemory) >= self.sizeState:
            #flat_list = [item for rij in self.__getDistanceMatrix__(timeInPast=timeInPast) for item in rij if item != -1]
            lenQueryMem = len(self.queryMemory)
            flat_list = [np.linalg.norm(self.queryMemory[i]-self.queryMemory[j]) for i in range(lenQueryMem-self.sizeState-timeInPast,lenQueryMem-timeInPast) for j in range(lenQueryMem-self.sizeState-timeInPast,i)]
            
            predLabels = self.predictedLabels[-self.sizeState-timeInPast:len(self.predictedLabels)-timeInPast]
            retLabels = self.returnedLabels[-self.sizeState-timeInPast:len(self.returnedLabels)-timeInPast]
            compLabels = [1 if predLabels[i]==retLabels[i] else 0 for i in range(0,len(predLabels))]
            
            flat_list.extend(compLabels)
            
            # flat_list.extend(self.predictedLabels[-self.sizeState-timeInPast:len(self.predictedLabels)-timeInPast])
            # flat_list.extend(self.returnedLabels[-self.sizeState-timeInPast:len(self.returnedLabels)-timeInPast])
            
            return np.asarray(flat_list)
        else:
            return None
        
    def getLastStateRepresentation(self):
        return self.getStateRepresentation(timeInPast=1) if len(self.queryMemory)-1 >= self.sizeState else None
        
    
    def __getDistanceMatrix__(self,timeInPast=0):
        distanceMatrix = [[np.linalg.norm(self.queryMemory[i]-self.queryMemory[j]) if i>j else -1 for i in range(len(self.queryMemory)-self.sizeState-timeInPast,len(self.queryMemory)-timeInPast)] for j in range(len(self.queryMemory)-self.sizeState-timeInPast,len(self.queryMemory)-timeInPast)]
        return distanceMatrix
    
    def getAverageStepSize(self):
        distances = np.asarray([np.linalg.norm(self.queryMemory[i]-self.queryMemory[i+1]) for i in range(len(self.queryMemory)-1)])
        return np.mean(distances)

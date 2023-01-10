import numpy as np
import torch
from scipy.spatial import distance
from scipy.special import kl_div, rel_entr
from tensorflow.keras.models import Sequential, load_model
from models.simEncMNIST import mnistNet
from models.simEncCIFAR import cifarNet
import tensorflow as tf

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def l2(a,b):
    #L2 distance
    return np.linalg.norm(a-b)

def l2unit(center, perturbed):
    vec = (perturbed - center)/l2(perturbed, center)
    return center + vec

def KL(P,Q):
     epsilon = 1e-8

     # You may want to instead make copies to avoid changing the np arrays.
     P = P+epsilon
     Q = Q+epsilon

     divergence = np.sum(P*np.log(P/Q))
     return divergence
 
class Contrasts():
    def __init__(self):
        self.tensors = []
        self.labels = []
        
    def add(self, tensor, label):
        self.tensors.append(tensor)
        self.labels.append(label)

    def save(self):
        torch.save(self.tensors, './data/contrasts.pt')
        torch.save(self.labels, './data/labels.pt')

# similarityModel = load_model('models/SIMILARITYmodel.h5', compile=False)
# def getSimilarityEncoding(query):
#     # print(type(query))
#     # npt = query.numpy()
#     # tft = tf.convert_to_tensor(npt)
#     s = similarityModel.predict(query, steps=1)
#     # z =  similarityModel.predict(tft, steps=1)
#     # print(l2(s, z))
#     return s


class SimEnc():
    def __init__(self, simemb, dataset):
        self.dt = dataset
        if self.dt == 'mnist':
            # self.similarityModel = load_model('models/MNISTencoder.h5', compile=False)
            self.similarityModel = mnistNet()
            self.similarityModel.load_state_dict(torch.load('models/MNISTembedding.pt', map_location=device))
            self.similarityModel.eval()
        elif self.dt == 'cifar':
            # self.similarityModel = load_model('models/CIFARencoder.pt', compile=False)
            self.similarityModel = cifarNet()
            if simemb == 2:
                self.similarityModel.load_state_dict(torch.load('models/CIFARembedding_trs.pt', map_location=device))
            else:
                self.similarityModel.load_state_dict(torch.load('models/CIFARembedding.pt', map_location=device))
            self.similarityModel.eval()
        
    # def getSimilarityEncoding(self, query):
    #     #start = time.time()
    #     query = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(query),0),3)
    #     a = self.similarityModel.predict(query, steps=1)
    #     # print(a.shape)
    #     #print(time.time() - start)
    #     return a
    
    def getSimilarityEncoding(self, query):
        if self.dt == 'mnist':
            with torch.no_grad():
                a = torch.unsqueeze(torch.from_numpy(query), dim=0)
                a = self.similarityModel(torch.unsqueeze(a, dim=0)).detach().numpy()
        elif self.dt == 'cifar':
            with torch.no_grad():
                a = self.similarityModel(torch.unsqueeze(torch.from_numpy(query), dim=0)).detach().numpy()
        return a


class Chain():
    def __init__(self, nrQueues=3, sizeState=30, train=True, simemb=1, dataset='mnist'):
        self.simenc = SimEnc(simemb, dataset)
        self.dim = 28 if dataset=='mnist' else 32
        self.shape = [28,28] if dataset=='mnist' else [3,32,32]
        self.sizeState = sizeState
        self.train = train
        self.nrQueues = nrQueues
        
    def reset(self):
        self.queues = []        
        self.adds = 0
        self.switch = True
        for i in range(self.nrQueues):
            self.queues.append(Queue(self.sizeState,self.dim,self.shape))
        # MNIST values    
        self.queues[0].stepsize.append(0.2)
        self.queues[1].stepsize.append(10)
        self.queues[2].stepsize.append(5)
        
    def checkQuery(self, query, logits):
        self.query = query
        self.logits = logits
        self.simEnc = self.simenc.getSimilarityEncoding(query)
        # encs = [l2(enc, simEnc) for enc in self.encs]
        # # if any(encs) < self.threshold:
        # print(encs)
        # self.encs.append(simEnc)
        # a = []
        # for i, queue in enumerate(self.queues):
        #     a.append(l2(queue.getEncoding(), simEnc))
        # # mb online update of threshold
        # if a[np.argmin(a)] < self.threshold:
        #     index = np.argmin(a)
        # else:
        #     index = 1
        if self.adds < 3:
            span = [0, 0, 0]
        else:
            x = l2(self.queues[0].getEncoding(-1), self.simEnc)
            y = l2(self.queues[0].getEncoding(-2), self.simEnc)
            z = l2(self.queues[1].getEncoding(-1), self.simEnc)
            # z = l2(self.queues[0].getEncoding(-3), self.simEnc)
            span = [x, y, z]
            # sp = [l2(self.queues[0].queryMemory[-1], self.query)]
            # print(sp)
        # Initialize start and origin based on the first query seen
        if self.adds == 0:
            self.queues[2].addQueryToQueue(self.query, self.logits, self.simEnc)
            start, origin = self.queues[2].getStartOrigin()
            self.queues[0].start = start
            self.queues[0].origin = origin
            self.queues[0].fQuery = query
            self.queues[1].start = origin
            self.queues[1].origin = start
            self.queues[1].fQuery = query    
        # if self.adds > 2:
        #     a = l2(self.queues[2].queryMemory[-1], self.queues[2].queryMemory[-2])
        #     b = l2(self.queues[2].getEncoding(-1), self.queues[2].getEncoding(-2))
        #     # print(a,"||",b)
        return span
    
    def addQuery(self, itwas):
        if itwas and self.adds >= 3:
            self.queues[1].addQueryToQueue(self.query, self.logits, self.simEnc)
            alt = self.getOriginQueue(1)
        else:
            self.queues[0].addQueryToQueue(self.query, self.logits, self.simEnc)
            alt = self.getOriginQueue(0)
        self.adds += 1
        return alt

    def retState(self):
        return self.queues[0].getStateRepresentation6(self.simEnc, self.query, self.logits.squeeze().numpy())                 
    
    def getState(self, index):
        # return self.queues[index].getStateRepresentation3()
        return self.queues[0].getStateRepresentation9(self.simEnc, self.query, self.logits.squeeze().numpy())
        # a = self.queues[0].getStateRepresentation7(self.simEnc, self.query, self.logits.squeeze().numpy())
        # b = self.queues[1].getStateRepresentation7(self.simEnc, self.query, self.logits.squeeze().numpy())
        # return np.asarray(np.concatenate((a, b), axis=None))
               
    def getStepSizeQueue(self,queueNumber):
        return self.queues[queueNumber].getStepSize()
    
    def getOriginQueue(self,queueNumber):
        return self.queues[queueNumber].getOrigin()
    
    def getLastStepQueue(self,queueNumber):
        return self.queues[queueNumber].getLastStep()
    
    def getQueueIsFull(self,queueNumber):
        return self.queues[queueNumber].getIfQueueIsFull()
    
    def getLastStateRepresentationQueue(self,queueNumber):
        return self.queues[queueNumber].getLastStateRepresentation()
    
    
# class Queues():
#     def __init__(self, nrQueues=2, sizeState=30, threshold=0.3, dataset='mnist'):
#         self.simenc = SimEnc(dataset)
#         self.dim = 28 if dataset=='mnist' else 32
#         # self.encs = deque(maxlen=history)
#         # self.encs.append(0)
#         self.queues = []        
#         self.threshold = threshold
#         self.sizeState = sizeState        
#         self.nrQueues = nrQueues
#         self.adds = 0
#         self.lastQueueUsed = [i for i in range(nrQueues)]
#         for i in range(nrQueues):
#             self.queues.append(Queue(self.sizeState, self.dim))
#         # MNIST values    
#         self.queues[0].stepsize.append(0.2)
#         self.queues[1].stepsize.append(10)
            
#     # assign to queue based on minimum similarity encoding
#     def addQuery(self, query, logits):
#         # torch.tensor(query).unsqueeze(0).unsqueeze(3)
#         # print(type(query))
#         simEnc = self.simenc.getSimilarityEncoding(query)
#         # encs = [l2(enc, simEnc) for enc in self.encs]
#         # # if any(encs) < self.threshold:
#         # print(encs)
#         # self.encs.append(simEnc)
#         # a = []
#         # for i, queue in enumerate(self.queues):
#         #     a.append(l2(queue.getEncoding(), simEnc))
#         # # mb online update of threshold
#         # if a[np.argmin(a)] < self.threshold:
#         #     index = np.argmin(a)
#         # else:
#         #     index = 1
#         if self.adds == 0:
#             index = 0
#         elif self.adds == 1:
#             index = 1
#         elif self.adds == 2:
#             index = 0
#         else:
#             x = l2(self.queues[0].getEncoding(-1), simEnc)
#             y = l2(self.queues[0].getEncoding(-2), simEnc)
#             z = l2(self.queues[1].getEncoding(-1), simEnc)
#             index = 0 if x < self.threshold or y < self.threshold or z < self.threshold else 1
#             # index = 0 if t < self.threshold else 1
#             # print(x,y,z, index)
#         self.queues[index].addQueryToQueue(query, logits, simEnc)
#         self.adds += 1
#         # if self.adds > 2:
#         #     print(l2(self.queues[index].encodings[-2], simEnc))
#         return index

#     def getState(self, index):
#         return self.queues[index].getStateRepresentation()
               
#     def getStepSizeQueue(self,queueNumber):
#         # print(queueNumber)
#         return self.queues[queueNumber].getStepSize()
    
#     def getOriginQueue(self,queueNumber):
#         return self.queues[queueNumber].getOrigin()
    
#     def getLastStepQueue(self,queueNumber):
#         return self.queues[queueNumber].getLastStep()
    
#     def getQueueIsFull(self,queueNumber):
#         return self.queues[queueNumber].getIfQueueIsFull()
    
#     def getLastStateRepresentationQueue(self,queueNumber):
#         return self.queues[queueNumber].getLastStateRepresentation()

class Queue():
    def __init__(self, sizeState, dim, shape):
        self.sizeState = sizeState
        self.dim = dim
        self.shape = shape
        self.sizeQueueMemory = 30
        self.amountOfQueries = 0
        self.encodings = []
        self.cosines = []
        self.queryMemory = []
        self.stepsize = []
        self.logits = []
        self.starts = []
        self.origins = []
        self.vecs = []
        self.maxstep = 0.1
        self.cos = 0
        self.encodings.append(np.ones((1,256)))
        self.encodings.append(np.ones((1,256))/2)
        self.cosines.append(0)
        # if self.dim == 28:
        # print(*shape)
        self.fQuery = np.ones((shape))
        self.queryMemory.append(np.ones((shape)))
        self.queryMemory.append(np.ones((shape))/2)
        # else:
        #     self.fQuery = np.ones((3,self.dim,self.dim))
        #     self.queryMemory.append(np.ones((3,self.dim,self.dim)))
        #     self.queryMemory.append(np.ones((3,self.dim,self.dim))/2)

    def addQueryToQueue(self, query, logit, similaritySpaceEncoding):
        # self.lastBenignLabel = realLabel
        if self.amountOfQueries >= self.sizeQueueMemory-1:
            self.queryMemory.pop(0)
            self.encodings.pop(0)
            self.vecs.pop(0)
            self.cosines.pop(0)
            self.stepsize.pop(0)
            self.logits.pop(0)
        # if not self.stepsize:
        #     # average on MNIST
        #     self.stepsize.append(0.5)
        # else:
        #    self.stepsize.append(l2(self.queryMemory[-1], query))
        if self.queryMemory:
            self.stepsize.append(round(l2(self.queryMemory[-1], query), 4))
        self.cosines.append(self.cos)
        self.encodings.append(similaritySpaceEncoding)
        self.logits.append(logit.squeeze().numpy())
        if len(self.queryMemory) > 0:
            self.fQuery = 0.95*self.fQuery + 0.05*query
            a = query.flatten() - self.fQuery.flatten()
            self.vecs.append(a)
        
        self.queryMemory.append(query)
        self.amountOfQueries += 1
        
        if self.amountOfQueries <= 100:
            rank = np.argsort(self.logits[-1])
            # print(rank)
            self.starts.append(rank[-1])
            self.origins.append(rank[-2])
            self.setStartOrigin()
            # self.maxstep = np.mean(self.stepsize)
            # print('rank1:', rank[-1])
            # print('rank2:', rank[-2])
            # print('start:', self.start)
            # print('origin:', self.origin)
        
        # if len(self.queryMemory) > 1:
        #     print(l2(self.queryMemory[-1], self.queryMemory[-2]))
        # print(self.stepsize[-1])
        
    def setStartOrigin(self):
        self.start = np.argmax(np.bincount(self.starts))
        self.origin = np.argmax(np.bincount(self.origins))
        
    def getOrigin(self):
        rank = np.argsort(self.logits[-1])
        if rank[-2] == self.start:
            return rank[-1]
        else:
            return rank[-2]
        
    def getStartOrigin(self):
        return self.start, self.origin
        
    def getEncoding(self, n):
        # if len(self.encodings) == 0:
        #     return 0
        return self.encodings[n]
    
    def getLastQuery(self):
        if len(self.queryMemory) == 0:
            return 0
        return self.queryMemory[-1]
    
    def getLastStep(self):
        return self.stepsize[-1], self.maxstep
    
    def getMaxstep(self):
        return self.maxstep
    
    def getStepSize(self):
        return self.stepsize
    
    def getIfQueueIsFull(self):
        return len(self.queryMemory) >= self.sizeState
    
    def getStateRepresentation(self):
        # print(len(self.logits))
        arr = []
        for i in range(len(self.logits)-1):
            arr.append(self.logits[-1] - self.logits[i])
        arr.append(self.logits[-1])
        if arr: arr = np.concatenate(np.array(arr))
        # arr = np.concatenate(self.logits)
        # print(arr.shape)
        state = np.zeros(300)
        state[0:len(arr)] = arr
        # print(state)
        return np.asarray(state)
    
    def getStateRepresentation2(self):
        lenQueryMem = len(self.queryMemory)
        low = max(lenQueryMem - self.sizeState, 0)
        high = max(lenQueryMem, 0)
        # Calculate all pairwise l2 distances
        # TODO: add cosine similarity as a proxy for perpendicular moves
        # consider simplified distances (last to previous 29)
        # flat_list = [l2(self.queryMemory[i], self.queryMemory[j]) \
        #                 for i in range(low, high) for j in range(low, i)]
        flat_list = [l2(self.encodings[i], self.encodings[j]) \
                        for i in range(low, high) for j in range(low, i)]
        state = np.zeros(465)
        state[0:len(flat_list)] = flat_list
        
        # print(np.asarray(state).shape)
        return np.asarray(state)
    
    def getStateRepresentation3(self):
        arr = []
        for i in range(len(self.encodings)-1):
            arr.append(l2(self.encodings[-1], self.encodings[i]))
        # print(len(self.encodings))
        # if arr: arr = np.concatenate(np.array(arr))
        state = np.ones(100)
        state[0:len(arr)] = arr
        
        # print(np.asarray(state).shape)
        # print(np.sort(np.asarray(state))[0:6])
        return np.sort(np.asarray(state))
    
    def getStateRepresentation4(self, enc):
        arr = []
        for i in range(len(self.encodings)-1):
            arr.append(l2(enc, self.encodings[-i-1]))
        # print(len(self.encodings))
        # if arr: arr = np.concatenate(np.array(arr))
        state = np.zeros(30)
        state[0:len(arr)] = arr
        
        # print(np.asarray(state).shape)
        # print(np.asarray(state)[0:6])
        # print(np.around(np.asarray(self.logits)[-6:-1], decimals=4))
        # a = np.asarray(state)
        # print(len(a[np.where(a>0.1)]))
        return np.asarray(state)
        
    def getStateRepresentation5(self, enc):
        arr = []
        # for i in range(len(self.encodings)-1):
        #     arr.append(l2(enc, self.encodings[-i-1]))
        # print(l2(enc, self.encodings[-1]))
        # print(l2(img, self.queryMemory[-1]))rank = np.argsort(logits)
        # print(len(self.encodings))
        for i in range(len(self.encodings)-1):
            # print(enc.shape, self.encodings[-i-1].shape)
            arr.append(1 - distance.cosine(enc, self.encodings[-i-1]))
        # for i in range(len(self.encodings)-1):
        #     arr.append(l2(enc, self.encodings[-i-1]))
        # print(len(self.encodings))
        # if arr: arr = np.concatenate(np.array(arr))
        state = np.zeros(30)
        state[0:len(arr)] = arr
        
        # print(np.asarray(state).shape)
        print(np.asarray(state)[0:6])
        # print(np.around(np.asarray(self.logits)[-6:-1], decimals=4))
        # a = np.asarray(state)
        # print(len(a[np.where(a>0.1)]))
        return np.asarray(state)
    
    def getStateRepresentation6(self, enc, query, logits):
        ar1 = []
        ar2 = []
        ar3 = []
        ar4 = []
        sz = min(len(self.encodings)-1, 25)
        for i in range(sz):
            ar1.append(l2(enc, self.encodings[-i-1]))
        # dist = round(l2(query, self.queryMemory[-1]), 4)
        # arr.append(dist)
        # arr.extend(np.flip(self.stepsize[1:]))
        # arr = [x / 10 for x in arr]
        # self.cos = distance.cosine((enc - self.encodings[-1]), (self.encodings[-1] - self.encodings[-2]))
        # qr = l2unit(self.queryMemory[-1], query)
        qr = query
        a = qr.flatten() - self.queryMemory[-1].flatten()
        # cos = distance.cosine(a, (self.queryMemory[-1].flatten() - self.queryMemory[-2].flatten()))
        # self.cos = round(cos, 4)
        
        # lg = min(len(self.logits), 25)
        for i in range(sz):
            # b = l2unit(self.queryMemory[-1], self.queryMemory[-i-2])
            b = self.queryMemory[-i-2]
            if not a.any() or not b.any():
                cos = 1
            else:
                cos = abs(1-distance.cosine(a, (b.flatten() - self.queryMemory[-1].flatten())))
            # cos = distance.cosine(a, (qr.flatten() - self.queryMemory[-i-2].flatten()))
            # cos = round(cos, 4)
            if i == 0: self.cos == cos
            # cos = sum(kl_div(logits, self.logits[-i-1]))
            ar2.append(cos)
        
        # ar2.append(self.cos)
        # ar2.extend(np.flip(self.cosines[1:]))

        # logit diff
        # ln = len(self.logits)
        lg = min(len(self.logits), 25)
        # if lg > 0: rank = np.argsort(logits)
        for i in range(lg):
            # logdif1 = logits[rank[-1]] - self.logits[-i-1][rank[-2]]
            # logdif1 = logits[rank[-1]]
            # logdif2 = logits[rank[-2]]
            # logdif2 = logits[rank[-2]] - self.logits[-i-1][rank[-1]]
            # logdif1 = abs(1-distance.cosine(logits, self.logits[-i-1]))
            # kldiv = sum(kl_div(logits, self.logits[-i-1]))
            kldiv = KL(logits, self.logits[-i-1])
            # if i == 0: print("DIV:", kldiv, logits, self.logits[-i-1])
            # kldiv = sum(rel_entr(logits, self.logits[-i-1]))
            # print(kldiv, sum(kldiv1))
            ar3.append(kldiv/20)
            # ar3.append(logdif1)
            # ar3.append(logdif2)

        rank = np.argsort(logits)
        # print(rank, self.start, self.origin, logits[rank[-1]], logits[rank[-2]])
        # print(rank, self.start, self.origin, rank[-1], rank[-2])
        ar4.append(logits[rank[-1]])
        ar4.append(logits[rank[-2]])

        state1 = np.zeros(25)
        # state1 = np.random.choice(ar1, size=25)
        state2 = np.ones(25)*0.5
        # state2 = np.random.choice(ar2, size=25)
        state3 = np.zeros(25)
        # state3 = np.random.choice(ar3, size=10) if lg else np.ones(10)
        state1[0:len(ar1)] = np.round(ar1, 5)
        state2[0:len(ar2)] = np.round(ar2, 5)
        state3[0:len(ar3)] = np.round(ar3, 5)
        state4 = np.round(ar4, 5)
        # state3 = np.round(state3, 4)
        # print(len(ar1), len(ar2))
        
        
        # print("STATE", np.asarray(state1)[0:3], np.asarray(state2)[0:3], np.asarray(state3)[0:3])
        # print(np.asarray(state).shape)
        # print(np.around(np.asarray(self.logits)[-6:-1], decimals=4))
        # a = np.asarray(state)
        # print(len(a[np.where(a>0.1)]))
        return np.asarray(np.concatenate((state1, state2, state3, state4), axis=None))
    
    def getStateRepresentation7(self, enc, query, logits):
        ar1 = []
        ar2 = []
        ar3 = []
        sz = min(len(self.encodings)-1, 25)
        for i in range(sz):
            ar1.append(l2(enc, self.encodings[-i-1]))
        qr = query
        a = qr.flatten() - self.queryMemory[-1].flatten()
        
        for i in range(sz):
            b = self.queryMemory[-i-2]
            cos = abs(1-distance.cosine(a, (b.flatten() - self.queryMemory[-1].flatten())))
            if i == 0: self.cos == cos
            ar2.append(cos)
        
        # lg = min(len(self.logits), 4)
        # if lg > 0: rank = np.argsort(logits)
        rank = np.argsort(logits)
        # print(rank, self.start, self.origin, logits[rank[-1]], logits[rank[-2]])
        ar3.append(logits[rank[-1]])
        ar3.append(logits[rank[-2]])
        # start = np.zeros(10)
        # start[self.start] = 1
        # origin = np.zeros(10)
        # origin[self.origin] = 1
        # cos = abs(1-distance.cosine(logits, self.logits[-i-1]))
        if rank[-1] == self.start:
            ar3.append(1)
        else:
            ar3.append(0)
        if rank[-2] == self.origin:
            ar3.append(1)
        else:
            ar3.append(0)

        # state1 = np.zeros(25)
        state1 = np.random.choice(ar1, size=25)
        # state2 = np.ones(25)
        state2 = np.random.choice(ar2, size=25)
        state3 = np.random.choice(ar3, size=4)
        state1[0:len(ar1)] = np.round(ar1, 5)
        state2[0:len(ar2)] = np.round(ar2, 5)
        state3[0:len(ar3)] = np.round(ar3, 5)
        # state3 = np.round(state3, 4)
        # print(len(ar1), len(ar2))
        
        # print("ETAT", np.asarray(state1)[0:6], np.asarray(state2)[0:6], np.asarray(state3))
        # print(np.asarray(state).shape)
        # print(np.around(np.asarray(self.logits)[-6:-1], decimals=4))
        # a = np.asarray(state)
        # print(len(a[np.where(a>0.1)]))
        return np.asarray(np.concatenate((state1, state2, state3), axis=None))
    
    def getStateRepresentation8(self, enc, query, logits):
        ar1 = []
        ar2 = []
        ar3 = []
        ar4 = []
        ar5 = []
        sz = min(len(self.encodings)-1, 25)
        for i in range(sz):
            ar1.append(l2(enc, self.encodings[-i-1]))

        qr = query
        a = qr.flatten() - self.queryMemory[-1].flatten()
        # cos = distance.cosine(a, (self.queryMemory[-1].flatten() - self.queryMemory[-2].flatten()))
        # self.cos = round(cos, 4)
        
        # lg = min(len(self.logits), 25)
        for i in range(sz):
            # b = l2unit(self.queryMemory[-1], self.queryMemory[-i-2])
            b = self.queryMemory[-i-2]
            if not a.any() or not b.any():
                cos = 1
            else:
                cos = abs(1-distance.cosine(a, (b.flatten() - self.queryMemory[-1].flatten())))
            # cos = distance.cosine(a, (qr.flatten() - self.queryMemory[-i-2].flatten()))
            # cos = round(cos, 4)
            if i == 0: self.cos == cos
            # cos = sum(kl_div(logits, self.logits[-i-1]))
            ar2.append(cos)

        qr = query
        a = qr.flatten() - self.fQuery.flatten() #+ np.random.uniform(0, 1e-5)
        # print(qr.flatten().sum(), self.fQuery.flatten().sum())
        vc = min(len(self.vecs)+1, 27)
        
        # print(len(self.vecs), vc)
        for i in range(vc-2):
            if vc < 3:
                cos = 0
            else:
                b = self.vecs[-i-1]
                # print(not a.any(), not b.any())
                if not a.any() or not b.any():
                    cos = 1
                    # print(b)
                else:
                    # print('xeption', i, vc)
                    cos = abs(1-distance.cosine(a, b))
            if i == 0: self.cos == cos
            ar3.append(cos)
        
        lg = min(len(self.logits), 25)
        # if lg > 0: rank = np.argsort(logits)
        for i in range(lg):
            kldiv = KL(logits, self.logits[-i-1])
            ar4.append(kldiv/20)

        rank = np.argsort(logits)
        # print(rank, self.start, self.origin, logits[rank[-1]], logits[rank[-2]])
        # print(rank, self.start, self.origin, rank[-1], rank[-2])
        ar5.append(logits[rank[-1]])
        ar5.append(logits[rank[-2]])

        state1 = np.zeros(25)
        state2 = np.ones(25)*0.5
        state3 = np.ones(25)*0.5
        state4 = np.zeros(25)
        state1[0:len(ar1)] = np.round(ar1, 5)
        state2[0:len(ar2)] = np.round(ar2, 5)
        state3[0:len(ar3)] = np.round(ar3, 5)
        state4[0:len(ar4)] = np.round(ar4, 5)
        state5 = np.round(ar5, 5)
        
        x = torch.empty(size=(25,self.dim,self.dim))
        rg = len(self.queryMemory)
        if rg > 24:
            for i in range(25):
                x[i] = torch.from_numpy(query - self.queryMemory[-i])
        else:
            for i in range(rg):
                x[i] = torch.from_numpy(query - self.queryMemory[-i])
            for i in range(25-rg):
                x[i+rg] = torch.from_numpy(query - (np.ones((self.dim,self.dim))/2))
            
        return np.asarray(np.concatenate((sum(state1), sum(state2), sum(state3), sum(state4), state5), axis=None)), x
    
    def getStateRepresentation9(self, enc, query, logits):
        x = torch.empty(size=(25,*self.shape))
        rg = len(self.queryMemory)
        if rg > 24:
            for i in range(25):
                x[i] = torch.from_numpy(query - self.queryMemory[-i])
        else:
            for i in range(rg):
                x[i] = torch.from_numpy(query - self.queryMemory[-i])
            for i in range(25-rg):
                x[i+rg] = torch.from_numpy(query - (np.ones((self.dim,self.dim))/2))
            
        return x, x
    
    def getLastStateRepresentation(self):
        return self.getStateRepresentation(timeInPast=1) if len(self.queryMemory)-1 >= self.sizeState else None
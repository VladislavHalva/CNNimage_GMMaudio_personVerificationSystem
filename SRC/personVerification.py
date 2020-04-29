import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io
from skimage.transform import rotate, AffineTransform, warp
import random
from skimage import img_as_ubyte
import os
from skimage.util import random_noise
import re
import librosa
from os.path import isfile, join, basename
import sys
import scipy.fftpack
from glob import glob
from scipy.io import wavfile
from scipy.special import logsumexp
from tqdm import tqdm
import pickle

pictureDirName = "pictures"
class dataAugmentation:
  def __init__(self,classification,dataRdy):
    self.classification = classification
    self.dataReady = dataRdy
    self.saveClassification ="trainingDataClassification.npy"
    self.saveVerification ="trainingDataVerification.npy"  
    
    self.shapeTransformation = 6+1
    self.shapeTransformationDict = {
        1 : self.flipImage,
        2 : self.leftRotate,
        3 : self.moreLeftRotate,
        4 : self.rightRotate,
        5 : self.moreRightRotate,
        6 : self.nothing,
    }

    self.colorTransformation = 6
    self.colorTransformationDict = {
        1 : self.randomNoise,
        2 : self.blurryImg,
        3 : self.gaussNoise,
        4 : self.addContras,
        5 : self.addMoreContrast,
        6 : self.addBrightness,
    }
    self.pictureToClassDict = {
        "m429" : 0,
        "m425" : 1,
        "m424" : 2,
        "m423" : 3,
        "m422" : 4,
        "m421" : 5,
        "m420" : 6,
        "m419" : 7,
        "m417" : 8,
        "m416" : 9,
        "m414" : 10,
        "f409" : 11,
        "f408" : 12,
        "f407" : 13,
        "f406" : 14,
        "f405" : 15,
        "f404" : 16,
        "f403" : 17,
        "f402" : 18,
        "f401" : 19,
    }

    self.pictureToVariDict = {
        "m429" : 0,
    }
  ### SHAPE TRANSFORMATION
  def flipImage(self,img):
    img = np.fliplr(img)
    return img/(np.amax(img))

  def leftRotate(self,img):
    return rotate(img,angle=random.randint(8,12))

  def rightRotate(self,img):
    return rotate(img,angle=-1 * random.randint(8,12))

  def moreLeftRotate(self,img):
    return rotate(img,angle=random.randint(15,20))

  def moreRightRotate(self,img):
    return rotate(img,angle=-1 * random.randint(15,20))

  def nothing(self,img):
    return img/(np.amax(img) )

  ### COLOR TRANSFORMATION
  def randomNoise(self,img,blank = 0.0):
    return random_noise(img)

  def blurryImg(self,img,blank = 0.0):
    return cv2.GaussianBlur(img,(5,5),0)

  def gaussNoise(self,img,variace = 0.0):
    var = 0.75
    gaussNoise = np.random.normal(0,var + var*variace,img.size)
    gaussNoise = gaussNoise.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    return cv2.add(img,gaussNoise)

  def addContras(self,img,alphaParam = 0.0):
    varAlpha = 1.5
    return cv2.convertScaleAbs(img,alpha=(varAlpha +varAlpha*alphaParam),beta=0)

  def addMoreContrast(self,img,alphaParam = 0.0):
    varAlpha = 2.0
    return cv2.convertScaleAbs(img,alpha=(varAlpha +varAlpha*alphaParam),beta=0)

  def addBrightness(self,img,betaParam = 0.0):
    varBeta = 50
    return cv2.convertScaleAbs(img,alpha=1.0,beta=(varBeta +varBeta*betaParam))

  def pictureAugmentation(self,img,pictureClass):
    if not self.classification:
      if(pictureClass == 0):
        augumentationData = self.randomAugmentation(img,2)
      else:
        augumentation = self.randomAugmentation(img)
        random.shuffle(augumentation)
        length = int(len(augumentation)/2.5)
        augumentationData = augumentation[:length]
    else:
      augumentationData = self.randomAugmentation(img)
      random.shuffle(augumentationData)
      if(pictureClass == 0):
        length = int(len(augumentationData)/2.5)
        augumentationData = augumentationData[:length]
    return augumentationData

  def randomAugmentation(self,img,size = 1):
    pictures = [img/255.0]
    randVal = 0.0
    for i in range(1,self.shapeTransformation):
      for j in range(0,size* self.colorTransformation):
        if(size == 2):
          randVal = random.uniform(0, 1) - 0.5
        colorTrans = self.colorTransformationDict[(j%6)+1](img,randVal)
        colorAndShapeTransform = self.shapeTransformationDict[i](colorTrans)
        pictures.append(colorAndShapeTransform)
    return pictures

  def findVerification(self,fileName):
    x = re.split("_.*$",fileName)
    return self.pictureToVariDict.setdefault(x[0], 1)

  def findClass(self,fileName):
    x = re.split("_.*$",fileName)
    return self.pictureToClassDict[x[0]]

  def prepareTrainingData(self):
    trainingData = []
    countClass = [0]*20
    allData = 0
    i = 0
    for fileName in os.listdir(pictureDirName):
        path = os.path.join(pictureDirName, fileName)
        img = cv2.imread(path)
        if self.classification:
          pictureClass = int(self.findClass(fileName))
        else:
          pictureClass = int(self.findVerification(fileName))
        augumentationData = self.pictureAugmentation(img,pictureClass)
        for pic in augumentationData:
          allData += 1
          countClass[pictureClass] += 1
          trainingData.append([np.array(pic),pictureClass])

    print("all data in set", allData)
    print("Count data class >> ", countClass)
    print("Index of array represent number of data per index class")
    np.random.shuffle(trainingData)
    if self.classification:
      saveFile = self.saveClassification
    else:
      saveFile = self.saveVerification  
    np.save(saveFile,trainingData)
  
  def prepareData(self):
    if not self.dataReady:
      self.prepareTrainingData()



class CNNet(nn.Module):
  def __init__(self,veri):
    super().__init__()
    self.verification = veri
    
    self.conv1 = nn.Conv2d(3, 16, 5)
    self.conv2 = nn.Conv2d(16, 32, 3)
    self.conv3 = nn.Conv2d(32, 64, 3)
    self.conv4 = nn.Conv2d(64, 128, 3)
    
    self.pool1 = nn.MaxPool2d(2, 2)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.pool3 = nn.MaxPool2d(2, 2)
    self.pool4 = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(3*3*128,512)
    self.fc2 = nn.Linear(512, 64)
    self.fc3 = nn.Linear(64,20)
    
    self.fc1v = nn.Linear(3*3*128, 256)
    self.fc2v = nn.Linear(256, 32)
    self.fc3v = nn.Linear(32, 2)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = F.relu(self.conv3(x))
    x = self.pool3(x)
    x = F.relu(self.conv4(x))
    x = self.pool4(x)
    x = x.view(x.size(0),-1)
    
    if not self.verification:
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
    else:
      x = F.relu(self.fc1v(x))
      x = F.relu(self.fc2v(x))
      x = self.fc3v(x)
    return x

  def getFineTuningParams(self):
    params = [
    {'params': self.fc1v.parameters()},
    {'params': self.fc2v.parameters()},
    {'params': self.fc3v.parameters()},
    ]
    return params

class CNNwrapper:
  def __init__(self,classification,epochs = 20, testRatio = 0.1,batchSize = 16):
    
    if(torch.cuda.is_available()):
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

    self.classification = classification
    self.CNN = CNNet(not classification)
    self.CNN.to(self.device)
    self.epochs = epochs
    self.batchSize = batchSize
    self.dataFileClass = "trainingDataClassification.npy"
    self.dataFileVerification = "trainingDataVerification.npy"
    self.testRatio = testRatio
    self.dataAugmentation = dataAugmentation(True,False)

  def loadCNN(self, CNNFile="/content/drive/My Drive/SUR/CNN_face_verification_Epocha20"):
    if self.device == torch.device("cpu"):
      self.CNN.load_state_dict(torch.load(CNNFile, map_location=lambda storage, loc: storage))
    else:
      self.CNN.load_state_dict(torch.load(CNNFile))

  def trainCNN(self,finetuning,prepareData):
    if prepareData:
      self.prepareTrainingTestingData()
    
    if not finetuning:
      self.classification = True
      self.loadDataClassification()
    else:
      self.classification = False
      self.loadDataVerification()
    self.trainingCNN()  

  def testSample(self,pathImg):
    img = cv2.imread(pathImg)
    ImgArray = np.array(img)
    ImgTensor = torch.Tensor(ImgArray)
    ImgTensor = ImgTensor.permute(2,0,1).to(self.device)
    netOut = self.CNN(ImgTensor.view(1,3,80,80))[0]
    score = netOut[0].item() -1000
    predictClass = 0
    if score > 0:
      predictClass = 1
    return (predictClass,score)

  def prepareTrainingTestingData(self):
    self.dataAugmentation.prepareData()  
    self.dataAugmentation = dataAugmentation(False,False)
    self.dataAugmentation.prepareData()  
  
  def validationResults(self):
    correct = 0
    total = 0

    if self.classification:
      testData = self.testClassSet
      testGT = self.testClassGT
      self.CNN.verification = False
    else:
      testData = self.testVeriSet
      testGT = self.testVeriGT
      self.CNN.verification = True

    with torch.no_grad():
      for i in range(len(testData)):
        realClass = testGT[i]
        netOut = self.CNN(testData[i].view(1,3,80,80))[0]
        predictClass = torch.argmax(netOut)
        total += 1
        if predictClass == realClass:
          correct += 1
    return (100*round(correct/total,5))

  def trainingCNN(self):
    optimizer = optim.Adam(self.CNN.parameters(), lr=0.001)
    if torch.cuda.is_available():
      lossFunction = nn.CrossEntropyLoss().cuda()
    else:
      lossFunction = nn.CrossEntropyLoss()

    if self.classification:
      trainData = self.trainClassSet
      trainGT = self.trainClassGT
      self.CNN.verification = False 
    else:
      trainData = self.trainVeriSet
      trainGT = self.trainVeriGT
      optimizer = optim.Adam(self.CNN.getFineTuningParams(), lr=0.001)
      self.CNN.verification = True 

    for epoch in range(self.epochs):
      for i in range(0, len(trainData), self.batchSize):
        batch = trainData[i:i+self.batchSize]
        batchGT = torch.autograd.Variable(trainGT[i:i+self.batchSize]) 
        self.CNN.zero_grad()
        output = self.CNN(batch)
        loss = lossFunction(output,batchGT)
        loss.backward()
        optimizer.step()
        if i % (self.batchSize*100) == 0:
          validAccuracy = self.validationResults()
          print("epoche> ", epoch," Iter> ",i," loss> ",loss.item(), "validation accuracy> ",validAccuracy)


  def loadDataVerification(self):
    trainingVeriData = np.load(self.dataFileVerification, allow_pickle=True)

    dataVeriSetTensor = torch.Tensor([i[0] for i in trainingVeriData])
    dataVeriGTTensor = torch.Tensor([i[1] for i in trainingVeriData])

    dataVeriSetTensor = dataVeriSetTensor.permute(0,3,1,2)
    dataVeriGTTensor = dataVeriGTTensor.long()

    testVeriSize = int(len(dataVeriSetTensor)* self.testRatio)

    self.trainVeriSet = dataVeriSetTensor[:-testVeriSize].to(self.device)
    self.trainVeriGT = dataVeriGTTensor[:-testVeriSize].to(self.device)

    self.testVeriSet = dataVeriSetTensor[-testVeriSize:].to(self.device)
    self.testVeriGT = dataVeriGTTensor[-testVeriSize:].to(self.device)

    print("test set length> ",len(self.trainVeriSet))
    print("validation set length> ",len(self.testVeriSet))
  
  def loadDataClassification(self):
    trainingClassData = np.load(self.dataFileClass, allow_pickle=True)

    dataSetClassTensor = torch.Tensor([i[0] for i in trainingClassData])
    dataClassGTTensor = torch.Tensor([i[1] for i in trainingClassData])

    dataSetClassTensor = dataSetClassTensor.permute(0,3,1,2)
    dataClassGTTensor = dataClassGTTensor.long()

    classTestSize = int(len(dataSetClassTensor)* self.testRatio)

    self.trainClassSet = dataSetClassTensor[:-classTestSize].to(self.device)
    self.trainClassGT = dataClassGTTensor[:-classTestSize].to(self.device)

    self.testClassSet = dataSetClassTensor[-classTestSize:].to(self.device)
    self.testClassGT = dataClassGTTensor[-classTestSize:].to(self.device)

    print("train set length> ",len(self.trainClassSet))
    print("test set length> ",len(self.testClassSet))
  

'''Audio augmentations'''

def addNoise(data, noiseFactor=0.004):
  noise = np.random.normal(0,1,len(data))
  augData = data + noiseFactor * noise
  return augData

def shiftTime(data):
  sr = 16000
  return np.roll(data, int(sr/10))

# 0.0 < factor < 1.0
def stretchTime(data, factor=0.4):
  return librosa.effects.time_stretch(data, factor)

def noAugment(data):
  return data

def applyAudioAugmentation(sample, augmentedCopies=8):
  augmentedData = []
  # applicable augmentations
  augmentations = [noAugment, stretchTime, shiftTime, addNoise]

  sample = sample.astype(float)

  for i in range(augmentedCopies):
    # apply first
    augId = np.random.randint(0,len(augmentations))
    augmented = augmentations[augId](sample)
    
    # apply second
    augId = np.random.randint(0,len(augmentations))
    augmented = augmentations[augId](augmented)

    # add to results
    augmentedData.append(augmented.astype(np.int16))

  return augmentedData

'''MFCC features extraction -IKRlib'''

def mel_inv(x):
    return (np.exp(x/1127.)-1.)*700.

def mel(x):
    return 1127.*np.log(1. + x/700.)

def mel_filter_bank(nfft, nbands, fs, fstart=0, fend=None):
    if not fend:
        fend = 0.5 * fs

    cbin = np.round(mel_inv(np.linspace(mel(fstart), mel(fend), nbands + 2)) / fs * nfft).astype(int)
    mfb = np.zeros((nfft // 2 + 1, nbands))
    for ii in range(nbands):
        mfb[cbin[ii]:  cbin[ii+1]+1, ii] = np.linspace(0., 1., cbin[ii+1] - cbin[ii]   + 1)
        mfb[cbin[ii+1]:cbin[ii+2]+1, ii] = np.linspace(1., 0., cbin[ii+2] - cbin[ii+1] + 1)
    return mfb

def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def spectrogram(x, window, noverlap=None, nfft=None):
    if np.isscalar(window): window = np.hamming(window)
    if noverlap is None:    noverlap = window.size / 2
    if nfft     is None:    nfft     = window.size
    x = framing(x, window.size, window.size-noverlap)
    x = scipy.fftpack.fft(x*window, nfft)
    return x[:,:x.shape[1]//2+1]

def mfcc(s, window, noverlap, nfft, fs, nbanks, nceps):
    snrdb = 40
    noise = np.random.rand(s.shape[0])
    s = s + noise.dot(np.linalg.norm(s, 2)) / np.linalg.norm(noise, 2) / (10 ** (snrdb / 20))

    mfb = mel_filter_bank(nfft, nbanks, fs, 32)
    dct_mx = scipy.fftpack.idct(np.eye(nceps, nbanks), norm='ortho') # the same DCT as in matlab

    S = spectrogram(s, window, noverlap, nfft)
    return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T

'''Features improvement'''

def improveFeatures(features):
  # remove first 2 seconds --> silence
  features = features[190:]

  energyTreshold = 38
  cutOffset = 18
  aboveCountTreshold = 5
  energy = features[:,0]
  meanEnergy = energy.mean()

  aboveMean = 0
  for i in range(energy.shape[0]-1,-1,-1):
    if energy[i] >= meanEnergy:
      aboveMean += 1
      if aboveMean > aboveCountTreshold:
        breakMeanIdx = i+cutOffset
        if breakMeanIdx >= features.shape[0]:
          breakMeanIdx = features.shape[0] - 1
        break

  # remove silence at the end
  features[:breakMeanIdx]
  return features

'''Audio loading and transformation to MFCC features'''

def wave16KhzToMFCC(dir_name, augment=False, multiplyCountBy=8, gender="both"):
  features = {}
  print("Loading audio files from: ", dir_name)
  for f in tqdm(glob(dir_name + '/*.wav')):
    # gender choice based on filename
    if gender=="male" and basename(f)[0:1] == "f":
      continue
    if gender=="female" and basename(f)[0:1] == "m":
      continue
    
    # load
    rate, sample = wavfile.read(f)
    assert(rate == 16000)

    # augment
    if(augment):
      i = 0
      data = applyAudioAugmentation(sample, augmentedCopies=multiplyCountBy)
      for sample in data:
        fcs = mfcc(sample, 400, 240, 512, 16000, 23, 13)
        fcs = improveFeatures(fcs)
        features[f+"_aug"+str(i)] = fcs
        i += 1
    else:
      fcs = mfcc(sample, 400, 240, 512, 16000, 23, 13)
      fcs = improveFeatures(fcs)
      features[f] = fcs
  return features

def wave16KhzToMFCCsample(filePath):
    rate, sample = wavfile.read(filePath)
    assert(rate == 16000)
    
    features = mfcc(sample, 400, 240, 512, 16000, 23, 13)
    features = improveFeatures(features)
    return features 


class GMMclassifier:
  def __init__(self, cl1Components, cl2Components, cl1Posterior=0.5, cl2Posterior=0.5):
    self.cl1Components = cl1Components
    self.cl2Components = cl2Components
    self.cl1Posterior = cl1Posterior
    self.cl2Posterior = cl2Posterior

  # from IKRlib
  def logistic_sigmoid(self, a):
    return 1 / (1 + np.exp(-a))

  # from IKRlib
  def logpdf_gmm(self, x, ws, mus, covs):
    return logsumexp([np.log(w) + self.logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)
  # from IKRlib
  def train_gmm(self, x, ws, mus, covs):
    gamma = np.vstack([np.log(w) + self.logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    tll = logevidence.sum()
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x)/gammasum[:,np.newaxis]
    
    if covs[0].ndim == 1: # diagonal covariance matrices
      covs = gamma.dot(x**2)/gammasum[:,np.newaxis] - mus**2
    else:
      covs = np.array([(gamma[i]*x.T).dot(x)/gammasum[i] - mus[i][:, np.newaxis].dot(mus[[i]]) for i in range(len(ws))])        
    return ws, mus, covs, tll

  def logpdf_gauss(self, x, mu, cov):
    assert(mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
    x = np.atleast_2d(x) - mu
    if cov.ndim == 1:
        return -0.5*(len(mu)*np.log(2 * np.pi) + np.sum(np.log(cov)) + np.sum((x**2)/cov, axis=1))
    else:
        return -0.5*(len(mu)*np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(np.linalg.inv(cov)) * x, axis=1))

  def train(self, iterations, cl1Data, cl2Data, log=False):
    self.cl1Data = cl1Data
    self.cl2Data = cl2Data
    
    # first class
    MUs_cl1  = self.cl1Data[np.random.randint(1, len(self.cl1Data), self.cl1Components)] # means
    COVs_cl1 = [np.cov(self.cl1Data.T)] * self.cl1Components # covariances
    Ws_cl1  = np.ones(self.cl1Components) / self.cl1Components; # weights

    # second class
    MUs_cl2  = self.cl2Data[np.random.randint(1, len(self.cl2Data), self.cl2Components)] # means
    COVs_cl2 = [np.cov(self.cl2Data.T)] * self.cl2Components # covariances
    Ws_cl2  = np.ones(self.cl2Components) / self.cl2Components; # weights

    # EM alg GMM training
    print("Training GMM: ")
    for i in tqdm(range(iterations)):
      [Ws_cl1, MUs_cl1, COVs_cl1, TTL_cl1] = self.train_gmm(self.cl1Data, Ws_cl1, MUs_cl1, COVs_cl1); 
      [Ws_cl2, MUs_cl2, COVs_cl2, TTL_cl2] = self.train_gmm(self.cl2Data, Ws_cl2, MUs_cl2, COVs_cl2); 
      if(log == True):
        print('Iteration:', i, ' Total log-likelihood for class1:', TTL_cl1, 'for class2:', TTL_cl2)

    self.Ws_cl1 = Ws_cl1
    self.MUs_cl1 = MUs_cl1
    self.COVs_cl1 = COVs_cl1
    self.Ws_cl2 = Ws_cl2
    self.MUs_cl2 = MUs_cl2
    self.COVs_cl2 = COVs_cl2
    print("Finished training")

  def testDir(self, testData, testedClass=1, log=False):
    # multiple data
    stats = []
    for sample in testData:
        sampleStats = []
        ll_cl1 = self.logpdf_gmm(sample, self.Ws_cl1, self.MUs_cl1, self.COVs_cl1)
        ll_cl2 = self.logpdf_gmm(sample, self.Ws_cl2, self.MUs_cl2, self.COVs_cl2)
        if(testedClass == 1):
          sampleStats.append((sum(ll_cl1) + np.log(self.cl1Posterior)) - (sum(ll_cl2) + np.log(self.cl2Posterior)))
          sampleStats.append(self.logistic_sigmoid(ll_cl1.mean() + np.log(self.cl1Posterior) - ll_cl2.mean() - np.log(self.cl2Posterior)))
        else:
          sampleStats.append((sum(ll_cl2) + np.log(self.cl2Posterior)) - (sum(ll_cl1) + np.log(self.cl1Posterior)))
          sampleStats.append(self.logistic_sigmoid(ll_cl2.mean() + np.log(self.cl2Posterior) - ll_cl1.mean() - np.log(self.cl1Posterior)))
        if score[-1] > 0:
          sampleStats.append(1)
        else:
          sampleStats.append(0)
        stats.append(sampleStats)
    if(log == True):
      for i in range(len(stats)):
        print("Score: ", stats[i][0], ", posterior: ", stats[i][1], ", decision: ", stats[i][2])
    return stats

  def testSample(self, sample, filename="unspecified", testedClass=1, log=False):
    ll_cl1 = self.logpdf_gmm(sample, self.Ws_cl1, self.MUs_cl1, self.COVs_cl1)
    ll_cl2 = self.logpdf_gmm(sample, self.Ws_cl2, self.MUs_cl2, self.COVs_cl2)
    if testedClass == 1:
      score = (sum(ll_cl1) + np.log(self.cl1Posterior)) - (sum(ll_cl2) + np.log(self.cl2Posterior))
      posterior = self.logistic_sigmoid(ll_cl1.mean() + np.log(self.cl1Posterior) - ll_cl2.mean() - np.log(self.cl2Posterior))
    else:
      score = (sum(ll_cl2) + np.log(self.cl2Posterior)) - (sum(ll_cl1) + np.log(self.cl1Posterior))
      posterior = self.logistic_sigmoid(ll_cl2.mean() + np.log(self.cl2Posterior) - ll_cl1.mean() - np.log(self.cl1Posterior))
    if score > 0:
      decision = 1
    else:
      decision = 0

    if log == True:
      print("File: ", filename, ", score: ", score, ", posterior: ", posterior, ", decision: ", decision)
    
    return filename, score, decision

  def getParams(self):
    return (self.Ws_cl1, self.MUs_cl1, self.COVs_cl1, self.Ws_cl2, self.MUs_cl2, self.COVs_cl2)

  def initWithParams(self, Ws_cl1, MUs_cl1, COVs_cl1, Ws_cl2, MUs_cl2, COVs_cl2):
    assert(Ws_cl1.shape[0] == self.cl1Components)
    assert(Ws_cl2.shape[0] == self.cl2Components)
    
    self.Ws_cl1 = Ws_cl1
    self.MUs_cl1 = MUs_cl1
    self.COVs_cl1 = COVs_cl1
    self.Ws_cl2 = Ws_cl2
    self.MUs_cl2 = MUs_cl2
    self.COVs_cl2 = COVs_cl2

  def loadParamsFromFile(self, fileName):
    with open(fileName, 'rb') as f:
      params = pickle.load(f)
    self.initWithParams(*params)

  def storeParamsToFile(self, fileName):
    params = self.getParams()
    with open(fileName, 'wb') as f:
      pickle.dump(params, f)

class GMMwrapper:
  def __init__(self, trainClass1Dir, trainClass2Dir, cl1Components, cl2Components, cl1Posterior=0.5, cl2Posterior=0.5, augmentCl1=0, augmentCl2=0):
    self.trainClass1Dir = trainClass1Dir
    self.trainClass2Dir = trainClass2Dir
    self.augmentCl1 = augmentCl1
    self.augmentCl2 = augmentCl2
    self.loadedTrainData = False
    self.cl1Components = cl1Components
    self.cl2Components = cl2Components
    self.classifier = GMMclassifier(self.cl1Components, self.cl2Components, cl1Posterior, cl2Posterior)

  def loadPreparedTrainData(self, trainCl1, trainCl2):
    self.trainCl1 = trainCl1
    self.trainCl2 = trainCl2
    self.loadedTrainData = True
  
  def train(self, iterations, log=False):
    if not self.loadedTrainData:
      if self.augmentCl1 == 0:
        self.trainCl1 = list(wave16KhzToMFCC(self.trainClass1Dir).values())
      else:
        self.trainCl1 = list(wave16KhzToMFCC(self.trainClass1Dir, augment=True, multiplyCountBy=self.augmentCl1).values())

      if self.augmentCl2 == 0:
        self.trainCl2 = list(wave16KhzToMFCC(self.trainClass2Dir).values())
      else:
        self.trainCl2 = list(wave16KhzToMFCC(self.trainClass2Dir, augment=True, multiplyCountBy=self.augmentCl2).values())

    self.trainCl1 = np.vstack(trainCl1)
    self.trainCl2 = np.vstack(trainCl2)

    self.classifier.train(iterations, self.trainCl1, self.trainCl2, log)

  def testSample(self, filePath, testedClass=1, log=False):
    sample = wave16KhzToMFCCsample(filePath)
    filename, score, decision = self.classifier.testSample(sample, basename(filePath), testedClass=testedClass, log=log)
    return score, decision

  def getParams(self):
    return self.classifier.getParams()

  def initWithParams(self, Ws_cl1, MUs_cl1, COVs_cl1, Ws_cl2, MUs_cl2, COVs_cl2):
    self.classifier.initWithParams(Ws_cl1, MUs_cl1, COVs_cl1, Ws_cl2, MUs_cl2, COVs_cl2)

  def loadParamsFromFile(self, fileName):
    return self.classifier.loadParamsFromFile(fileName)

  def storeParamsToFile(self, fileName):
    self.classifier.storeParamsToFile(fileName)




class PersonVerification:
  def __init__(self, evalDir):   
    if(torch.cuda.is_available()):
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")
    self.dirName = evalDir
    self.CNN = CNNwrapper(True)
    self.CNN.loadCNN("../models/CNN_face_verification")
    self.GMM = GMMwrapper("","", 64,64)
    self.GMM.loadParamsFromFile("../models/GMM_params.pickle")
    self.resultFilePicture = "../picture_CNN.txt"
    self.resultFileAudio = "../audio_GMM.txt"
    self.resultFileCombined = "../combined_Picture_CNN_Audio_GMM.txt"

  def verificationPerson(self):
    resFilePicture = open(self.resultFilePicture,"w")
    resFileAudio = open(self.resultFileAudio,"w")
    resFileCombined = open(self.resultFileCombined,"w")

    print("Verification start with dir: ",self.dirName)
    for fileName in tqdm(os.listdir(self.dirName)):
      pathImg = os.path.join(self.dirName, fileName)
      reSearchResult = re.search("^.*png$", pathImg)
      if reSearchResult:
        pathAudio = pathImg[:-3] + "wav"
        probabilityImg, scoreImg = self.verificationCNN(pathImg)
        probabilityAudio, scoreAudio = self.verificationSoundGMM(pathAudio)
        self.preprocesAndSaveResultPicture(probabilityImg,scoreImg,resFilePicture,fileName)
        self.preprocesAndSaveResultAudio(probabilityAudio,scoreAudio,resFileAudio,fileName)
        self.preprocesAndSaveResultCombined(scoreImg,scoreAudio,resFileCombined,fileName)
    print("Verification done")
    resFilePicture.close()
    resFileAudio.close()
    resFileCombined.close()

  def preprocesAndSaveResultPicture(self,probabilityImg,scoreImg,saveFile,fileName):
    resultLine = fileName[:-4] +" "+str(scoreImg)+" " + str(probabilityImg)+"\n"
    saveFile.write(resultLine)

  def verificationCNN(self,pathImg):
    predictClass,score = self.CNN.testSample(pathImg)
    return (predictClass,score)

  def preprocesAndSaveResultAudio(self,probabilityAudio,scoreAudio,saveFile,fileName):
    fileName = fileName[:-3] + "wav"
    resultLine = fileName[:-4] +" "+str(scoreAudio)+" " + str(probabilityAudio)+"\n"
    saveFile.write(resultLine)
    
  def verificationSoundGMM(self,pathAudio):
    score, decision = self.GMM.testSample(pathAudio)
    return (decision, score)

  def verificationCombined(self, scorePicture, scoreAudio):
    if scorePicture + scoreAudio > 200:
      return 1, scorePicture + scoreAudio
    else:
      if scoreAudio > -55:
        return 1, scoreAudio
    return 0, scorePicture + scoreAudio

  def preprocesAndSaveResultCombined(self,scoreImg,scoreAudio,saveFile,fileName):
    probability, score = self.verificationCombined(scoreImg, scoreAudio)
    resultLine = fileName[:-4] +" "+str(score)+" " + str(probability)+"\n"
    saveFile.write(resultLine)


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Error, first argument has to be the evaluation data directory.")
  else:
    evalDataDir = sys.argv[1]

    personVeri = PersonVerification(evalDataDir)
    personVeri.verificationPerson()

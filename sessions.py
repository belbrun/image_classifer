
import numpy as np
from network import *
from layers import *
from datautil import *


class Session:

    def __init__(self):
        pass

    def start(self):
        pass

    def setNeuralNet(self, neuralNet):
        self.neuralNet = neuralNet

    def loadNeuralNet(self, path, loadConfig = True):
        self.neuralNet = NeuralNetwork.load(path, loadConfig)

    def getEntity(self, index, test):
        entity, label = datautil.getEntity(index, test)
        return (entity, self.convertLabel(label))

    def convertLabel(self, label):
        """
            Create a numpy array containing a correct classification result for
            a given label.
        """
        resultArray = np.zeros(10)
        resultArray[int(label)] = 1
        return resultArray

    def formatMessage(self, action, epoch, error, index = 0, output = None, isAvarage = False):

        message = ''
        message += action + '-- Epoch:' + str(epoch)
        message += ' Example: ' + str(index) if not isAvarage else \
            'AvarageError: ' + str(error)
        if output is not None:
            message += ' Output: ' + str(output[0])

        return message



class TrainingSession(Session):

    def __init__(self, datasetSize, trainingSetFactor, validationSetFactor,
            datasetPath, networkPath, learningRate = 0.01,
            drop = 1, startEpoch = 0, epochs = 0):
        self.startEpoch = startEpoch
        self.epochs = epochs
        self.drop = drop
        self.learningRate = learningRate
        self.trainingLog = []
        self.datasetPath = datasetPath
        self.networkPath = networkPath
        self.datasetSize = datasetSize
        self.trainingSetSize = int(round(datasetSize * trainingSetFactor))
        self.validationSetSize = int(round(datasetSize * validationSetFactor))

    def trainOnEntity(self, neuralNet, index, learningRate):
        entity, label = self.getEntity(index, False)
        return neuralNet.train(entity, label, learningRate)


    def feedEntity(self, neuralNet, index):
        entity, label = self.getEntity(index, False)
        return neuralNet.feedForError(entity, label)

    def train(self, epoch):
        avgError = 0
        for index in range(1, self.trainingSetSize):
            errors = self.trainOnEntity(self.neuralNet, index, self.learningRate)
            avgError += abs(errors)
            self.trainingLog.append(self.formatMessage('TRAINING', epoch, errors, index))
            print(self.trainingLog[-1])

        self.trainingLog.append(self.formatMessage('TRAINING', epoch, \
             avgError/(self.trainingSetSize), isAvarage = True))
        print(self.trainingLog[-1])

    def validate(self, epoch):

        avgError = 0
        results = []

        for index in range(self.trainingSetSize, self.trainingSetSize + self.validationSetSize):
            output, error = self.feedEntity(self.neuralNet, index)
            results.append(output)
            avgError += abs(error)
            self.trainingLog.append(self.formatMessage('VALIDATION', epoch, error, index, output))
            print(self.trainingLog[-1])

        #self.trainingLog.append\
        #('VALIDATION RESULTS : ' + str(results))
        #print(self.trainingLog[-1])
        self.trainingLog.append(self.formatMessage('VALIDATION', epoch, \
            avgError/(self.validationSetSize), isAvarage = True))
        print(self.trainingLog[-1])

    def process(self, epoch):

        self.saveEpoch(self.neuralNet, epoch, self.trainingLog)

    def saveEpoch(self, neuralNet, epoch, log = None, data = None):
        path = self.networkPath + 'epoch' + str(epoch) + '/'
        datautil.makeDirectory(path)
        neuralNet.save(path)
        if log:
            datautil.writeLog(path, log)

    def start(self):

        print('Training set size: ', self.trainingSetSize, 'validationSetSize: ', self.validationSetSize)

        #if self.startEpoch == 1:
        #    saveEpoch(self.neuralNet, 0, data = [str(self.gray), \
        #        str(self.avaraged), str(self.shape), \
        #        str(self.trainingSetFactor), str(self.validationSetFactor)])

        lastAvgError = None
        for i in range(self.startEpoch, self.epochs + 1):

            self.train(i)
            self.validate(i)
            self.process(i)
            self.learningRate *= self.drop

        return self.trainingLog

class TestingSession(Session):

    def __init__(self, numOfExamples):
        self.numOfExamples = numOfExamples


    def testEntity(self, index):
        entity, label = self.getEntity(index, test = True)
        return (self.neuralNet.classify(entity), label)

    def start(self):
        counts = [0,0] # correct, false
        for index in range(1, numOfExamples):
            (class, probability), label = self.testEntity(index)

            if class == label:
                counts[0] += 1
            else:
                counts[2] += 1
            print('TEST----Example: ', index, ' Label: ', label,
                        ' Class: ',class, ' Overall: ', counts)

        return [round(x/sum(counts)*100, 2) for x in counts]
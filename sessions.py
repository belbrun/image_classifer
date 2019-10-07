
import limitcalculator as calc
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
        return datautil.getEntity(index, test)


class TrainingSession(Session):

    def __init__(self, datasetSize, trainingSetFactor, validationSetFactor,
            datasetPath, learningRate = 0.01,
            drop = 1, startEpoch = 0, epochs = 0):
        self.startEpoch = startEpoch
        self.epochs = epochs
        self.drop = drop
        self.learningRate = learningRate
        self.trainingLog = []
        self.datasetPath = datasetPath
        self.datasetSize = datasetSize
        self.trainingSetSize = int(round(datasetSize * trainingSetFactor / 2))
        self.validationSetSize = int(round(datasetSize * validationSetFactor/2))

    def trainOnEntity(self, neuralNet, index, learningRate):
        entity, label = self.getEntity(index, False)
        return neuralNet.train(entity, label, learningRate)


    def feedEntity(self, neuralNet, index, isBlastom):
        entity, label = self.getEntity(index, False)
        return neuralNet.feedForError(entity, label, learningRate)

    def train(self):
        avgError = 0
        for index in range(1, self.trainingSetSize):
            errors = self.trainOnEntity(self.neuralNet, index, True, self.learningRate)
            avgError += abs(errors[r])
            self.trainingLog.append(formatMessage('TRAINING', i, errors[r], index))
            print(self.trainingLog[-1])

        self.trainingLog.append(formatMessage('TRAINING', i, \
             avgError/(trainingSetSize*2), isAvarage = True))
        print(self.trainingLog[-1])

    def validate(self):

        avgError = 0
        results = []

        for index in range(self.trainingSetSize, self.trainingSetSize + self.validationSetSize):
            output, error = self.feedEntity(neuralNet, index)
            results.append(output[0])
            avgError += abs(error)
            self.trainingLog.append(formatMessage('VALIDATION', i, error, index, output))
            print(self.trainingLog[-1])

    def process(self, epoch):
        currentAvgError =  avgError/(validationSetSize*2)
        self.trainingLog.append\
        ('VALIDATION RESULTS : ' + str(results) + 'LIMIT: ' + str(limit))
        print(self.trainingLog[-1])
        self.trainingLog.append(formatMessage('VALIDATION', i, \
            currentAvgError, isAvarage = True))
        print(self.trainingLog[-1])
        saveEpoch(neuralNet, epoch, self.trainingLog, \
        [str(gray), str(avaraged), str(shape), str(trainingSetFactor), \
        str(validationSetFactor)])

    def start(self):

        print('Training set size: ', self.trainingSetSize, 'validationSetSize: ', self.validationSetSize)

        if self.startEpoch == 1:
            saveEpoch(self.neuralNet, 0, data = [str(self.gray), \
                str(self.avaraged), str(self.shape), \
                str(self.trainingSetFactor), str(self.validationSetFactor)])

        lastAvgError = None
        for i in range(self.startEpoch, self.epochs + 1):

            self.train()
            self.validate()
            self.process(i)
            learningRate *= drop

        return self.trainingLog

class TestingSession(Session):

    def __init__(self, datasetPath, startIndex, endIndex, blastomCount, numOfEpochs = 1,
        gray = False, shape = None,  rotations = False, avaraged = False):
        self.datasetPath = datasetPath
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.blastomCount = blastomCount
        self.numOfEpochs = numOfEpochs
        self.gray = gray
        self.shape = shape
        self.rotations = rotations
        self.avaraged = avaraged

    def testEntity(self, index, isBlastom):
        index += 0 if isBlastom else self.blastomCount
        return self.neuralNet.classify(self.getEntity(index, isBlastom)[0])

    def start(self):
        counts = [0,0,0] #correct, false positives, false negatives
        for index in range(self.startIndex, self.endIndex + 1):
            correct = bool(self.testEntity(index, True))

            if correct:
                counts[0] += 1
            else:
                counts[2] += 1
            print('TEST----Example: ', index, '_1 Output: ', correct, ' Overall: ', counts)

            correct = not bool(self.testEntity(index, False))
            if correct:
                counts[0] += 1
            else:
                counts[1] += 1
            print('TEST----Example: ', index, '_0 Output: ', correct, ' Overall: ', counts)
        return [round(x/sum(counts)*100, 2) for x in counts]

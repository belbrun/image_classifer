from functions import *
from layers import *
import datautil

class NeuralNetwork():

    def __init__(self, errorFunction = simpleCost, classificationLimit = 0.5):
        self.errorFunction = errorFunction
        self.layers = []
        self.classificationLimit = classificationLimit


    def addLayer(self, layer):
        """
        Add a layer to the neural network in sequential fashion.
        """
        self.layers.append(layer)

    def output(self, x):
        """
            Calculate the output for a single input instance x (one row from
            the training or test set)
        """

        for layer in self.layers:
            x = layer.propagateForward(x)
            #print(layer, x)

        return x

    def calculateOutputs(self, data):
        outputs = []

        for instance in data:
            outputs.append(self.output(instance))

        return outputs

    def classify(self, x):
        return self.classifyResult(self.output(x)[0])

    def feed(self, data):
        outputs = self.calculateOutputs(data)
        return outputs

    def feedForError(self, x, result):
        output = self.output(x)
        error = self.calculateError(output, result)
        return (output, error)

    def calculateError(self, output, result):

        errors = []
        for i in range(0, result.shape[0]):
            errors.append(self.errorFunction(output[i], result[i]))

        return np.array(errors)

    def getOverallError(self, outputs, results):

        overallError = 0
        for i in range(0, len(outputs)):
            for error in self.calculateError(outputs[i], results[i]):
                overallError += error #absoulte ?

        return error

    def learn(self, errors, learningRate):

        for index,layer in enumerate(self.layers[::-1]):
            #print('----------------')
            #print(layer, '\n', errors)
            #learningRate *= 2
            errors = layer.propagateBackwards(errors, learningRate**(index-3))


    def train(self, x, results, learningRate):
        output = self.output(x)
        errors = self.calculateError(output, results)
        self.learn(errors, learningRate)
        return errors

    def calculateClassificationLimit(blastomResults, otherResults):
        blastomAvarage = sum(blastomResults)/len(blastomResults)
        otherAvarage = sum(otherResults)/len(blastomResults)
        difference = abs(blastomAvarage - otherAvarage)/100
        top = max([max(otherResults), max(blastomResults)])
        limit = min([min(otherResults), min(blastomResults)])
        print(top, limit)
        bestResults = [0,0,0]
        bestLimit = 0
        while limit < top:
            print('IN')
            correctBlastoms = len([i for i in blastomResults if i > limit])
            correctOthers = len([i for i in otherResults if i < limit])
            results = [correctBlastoms + correctOthers, \
                        len(otherResults) - correctOthers, \
                        len(blastomResults) - correctBlastoms]

            isBetterResult = results[0] > bestResults[0]
            hasLessFP = results[0] == bestResults[0] and results[1] < bestResults[1]

            if isBetterResult or hasLessFP:
                print(isBetterResult, hasLessFP)
                print(limit, results)
                bestLimit = limit
                bestResults = results

            limit += difference

        return (bestLimit, bestResults)

        #len([i for i in blastomResults if i > limit)


    def classifyResult(self, result):
        print(result, self.classificationLimit, result > self.classificationLimit)
        return int(result > self.classificationLimit)

    def setClassificationLimit(self, classificationLimit):
        self.classificationLimit = classificationLimit

    def save(self, path, data = None):
        if data:
            data.append(str(self.classificationLimit))
            datautil.saveData(path, data)

        for (index, layer) in enumerate(self.layers):
            newPath = path + str(index) + '/'
            datautil.makeDirectory(newPath)
            layer.save(newPath)

    def load(path, loadConfig = True):
        if loadConfig:
            classificationLimit = float(datautil.loadData(path)[-2])
        else :
            classificationLimit = 0.5
        network = NeuralNetwork(crossEntropyLoss, classificationLimit)
        for (index, id) in enumerate(datautil.getLayerIds(path)):
            network.addLayer(\
                getLayerById(id).load(path + str(index) + '/'))
        return network

    def printNetwork(self):
        for layer in self.layers:
            print(layer)
            if isinstance(layer, ConvolutionLayer):
                print(layer.filters)
                print(layer.bias)
            if isinstance(layer, FullyConnectedLayer):
                print(layer.weights)
                print(layer.bias)

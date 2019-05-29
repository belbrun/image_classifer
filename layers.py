import numpy as np
import datautil
from functions import getActivationFunction

minWeight = -0.33
maxWeight = 0.33

class Layer:

    def __init__(self, activationFunction):
        self.activationFunction = activationFunction


    def propagateForward(self, input):
        pass

    def propagateBackwards(self, errors):
        pass

    def save(self, path):
        pass

    def load(path):
        pass


class ConvolutionLayer(Layer):


    def __init__(self, filterNumber, filterSize, stride, activationFunction,\
                 inputDepth, filters = None, bias = None):

        if filters is not None:
            self.filters = filters
        else:
            self.filters = ConvolutionLayer\
                .initializeFilters(filterNumber, inputDepth, filterSize)
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.uniform(minWeight, maxWeight, size=filterNumber)

        self.filterNumber = filterNumber
        self.filterSize = filterSize
        self.stride = stride
        self.activationFunction = activationFunction
        self.data = None
        self.inputSize = 0
        self.inputDepth = inputDepth

    def initializeFilters(filterNumber, inputDepth, filterSize, path = None):

        filters = []
        for i in range(0, filterNumber):
            filterGroup = []
            for j in range(0, inputDepth):
                filter = None
                if path:
                    filter = np.load(path+str(i)+str(j)+'.npy')
                else:
                    filter = np.random.uniform(minWeight, maxWeight,size=(filterSize,filterSize))
                filterGroup.append(filter)
            filters.append(filterGroup)

        return filters




    def propagateForward(self, input):
        self.data = input
        self.inputSize = input[0].shape[0]
        self.z = []
        outputShape = int((self.inputSize - self.filterSize)/self.stride + 1)
        for (index,filterGroup) in enumerate(self.filters):

            self.z.append(np.zeros((outputShape,outputShape)))

            for i in range(0, outputShape):
                for j in range(0, outputShape):
                    for a in range(0, self.filterSize):
                        for b in range(0, self.filterSize):
                            for (d,x) in enumerate(input):

                                self.z[index][i,j] += filterGroup[d][a,b] * \
                                        x[i*self.stride + a, j*self.stride + b]

                    self.z[index][i,j] += self.bias[index]

        return self.activationFunction.activate(self.z)


    def propagateBackwards(self, errors, learningRate):

        weightErrors = [[]] * self.filterNumber
        previousErrors = []
        biasErrors = [0]* self.filterNumber
        outputShape = int((self.inputSize - self.filterSize)/self.stride + 1)

        for (d,x) in enumerate(self.data):

            previousErrors.append(np.zeros((self.inputSize,self.inputSize)))

            for (index, filterGroup) in enumerate(self.filters):

                weightErrors[index].append(np.zeros((self.filterSize,self.filterSize)))

                for i in range(0, outputShape):
                    for j in range(0, outputShape):

                        outputError = errors[index][i,j] * \
                            self.activationFunction.derived(self.z[index][i,j])

                        for a in range(0, self.filterSize):
                            for b in range(0, self.filterSize):

                                weightErrors[index][d][a,b] +=  \
                                    x[i+a, j+b] * outputError
                                previousErrors[d][i*self.stride +a, j*self.stride+b]\
                                    += outputError * self.filters[index][d][a,b]

                    biasErrors[index] += self.bias[index]*outputError

        self.correctWeights(weightErrors, biasErrors, learningRate)

        return previousErrors

    def correctWeights(self, weightErrors, biasErrors, learningRate):

        #print('Errors: ', weightErrors * int(learningRate))
        for (index, filterGroup) in enumerate(self.filters):
            for (d, filter) in enumerate(filterGroup):
                for a in range(0, self.filterSize):
                    for b in range(0, self.filterSize):
                        filter[a,b] += \
                            learningRate * weightErrors[index][d][a,b]

            self.bias[index] += learningRate * biasErrors[index]

    def save(self, path):
        datautil.saveData(path, ['CONV', self.filterNumber, self.filterSize,\
        self.stride, self.inputDepth, self.activationFunction.getName()])
        for (i, filterGroup) in enumerate(self.filters):
            for (j, filter) in enumerate(filterGroup):
                np.save(path + str(i) + str(j), filter)
        np.save(path + 'b', self.bias)

    def load(path):
        data = datautil.loadData(path)
        filterNumber = int(data[1])
        filterSize = int(data[2])
        stride = int(data[3])
        inputDepth = int(data[4])
        activationFunction = getActivationFunction(data[5])
        filters = ConvolutionLayer\
            .initializeFilters(filterNumber, inputDepth, filterSize, path)
        bias = np.load(path +'b.npy')
        return ConvolutionLayer(filterNumber, filterSize, stride, activationFunction,\
             inputDepth, filters, bias)


class MaxpoolLayer(Layer):

    def __init__(self, clusterSize = 2):
        self.clusterSize = clusterSize
        self.dataLength = 0


    def propagateForward(self, input):

        self.dataLength = len(input)
        self.inputSize = input[0].shape[0]
        outputShape = self.inputSize - self.clusterSize + 1
        self.z = []
        self.maxPositions = []


        for (d, data) in enumerate(input):

            self.z.append(np.empty((outputShape,outputShape)))
            self.maxPositions.append([])

            for i in range(0, self.inputSize - self.clusterSize + 1 ):
                for j in range(0, self.inputSize - self.clusterSize + 1):

                    max = data[i,j]
                    maxPosition = (i,j)

                    for k in range(0, self.clusterSize):
                        for l in range(0, self.clusterSize):

                            if(data[i+k, j+l] > max):
                                max = data[i+k, j+l]
                                maxPosition = (i+k, j+l)

                    self.maxPositions[d].append(maxPosition)
                    self.z[d][i, j] = max

        return self.z


    def propagateBackwards(self, errors, learningRate):

        previousErrors = []
        outputShape = self.inputSize - self.clusterSize + 1

        for d in range(0, self.dataLength):

            previousErrors.append(np.zeros((self.inputSize,self.inputSize)))

            for i in range(0, outputShape):
                for j in range(0, outputShape):

                    x, y = self.maxPositions[d][i * outputShape + j]
                    previousErrors[d][x,y] += errors[d][i,j]

        return previousErrors

    def save(self, path):
        datautil.saveData(path, ['MAXP', self.clusterSize])

    def load(path):
        data = datautil.loadData(path)
        return MaxpoolLayer(int(data[1]))


class FlatteningLayer(Layer):

    def __init__(self):
        self.inputSize = 0
        self.inputDepth = 0

    def propagateForward(self, input):

        self.inputDepth = len(input)
        self.inputSize = input[0].shape[0]

        flatLayers = []

        for layer in input:
            flatLayers.append(layer.flatten())

        return np.concatenate(input).ravel()

    def propagateBackwards(self, errors, learningRate):

        previousErrors = []
        layerLength = int(len(errors)/self.inputDepth)


        for layerIndex in range(0, self.inputDepth):
            layerErrors = \
                errors[layerIndex * layerLength: (layerIndex + 1) * layerLength]
            previousErrors.append\
                (np.resize(layerErrors,(self.inputSize, self.inputSize)))
        return previousErrors

    def save(self, path):
        datautil.saveData(path, ['FLAT'])

    def load(path):
        return FlatteningLayer()




class FullyConnectedLayer(Layer):

    def __init__(self, size, inputSize, activationFunction,  \
                    weights = None, bias = None, softmax = False):

        if weights is not None:
            self.weights = weights
        else :
            self.weights = FullyConnectedLayer.initializeWeights(size, inputSize)

        if bias is not None:
            self.bias = bias
        else :
            self.bias = np.random.uniform(minWeight, maxWeight,size=size)


        self.size = size
        self.inputSize = inputSize
        self.activationFunction = activationFunction
        self.softmax = softmax

    def initializeWeights(size, inputSize, path = None):

        weights = []
        for i in range(0, size):
            weightVector = None
            if path:
                weightVector = np.load(path + str(i) + '.npy')
            else :
                weightVector = np.random.uniform(minWeight, maxWeight,size=(inputSize,1))

            weights.append(weightVector)

        return weights

    def propagateForward(self, input):

        self.data = input
        self.z = np.empty(self.size)
        for i in range(self.size):
            self.z[i] = input.dot(self.weights[i]) + self.bias[i]

        return self.activationFunction.activate([self.z])[0]


    def propagateBackwards(self, errors, learningRate):

        weightErrors = []
        previousErrors = np.zeros(self.inputSize)
        biasErrors = [0] * self.size

        for neuronIndex in range(0, self.size):

            weightErrors.append(np.zeros(self.inputSize))
            gradientArg = (self.z, neuronIndex) if self.softmax else \
                            self.z[neuronIndex]
            outputError = errors[neuronIndex] * \
                self.activationFunction.derived(gradientArg)
            for i in range(self.inputSize):

                weightErrors[neuronIndex][i] += \
                    self.data[i] * outputError
                previousErrors[i] += \
                    self.weights[neuronIndex][i] * outputError

            biasErrors += self.bias[neuronIndex] * outputError

        self.correctWeights(weightErrors, biasErrors, learningRate)
        return previousErrors

    def correctWeights(self, weightErrors, biasErrors, learningRate):

            #print('Errors: ', weightErrors * int(learningRate))
            for neuronIndex in range(0, self.size):

                for i in range(0, self.inputSize):

                    self.weights[neuronIndex][i] += \
                        learningRate * weightErrors[neuronIndex][i]
                self.bias[neuronIndex] += \
                    learningRate * biasErrors[neuronIndex]

    def save(self, path):
        datautil.saveData(path, ['FCON', self.size, self.inputSize, \
            self.activationFunction.getName(), int(self.softmax)])

        for (i, weight) in enumerate(self.weights):
            np.save(path + str(i), weight)

        np.save(path + 'b', self.bias)

    def load(path):
        data = datautil.loadData(path)
        size = int(data[1])
        inputSize = int(data[2])
        activationFunction = getActivationFunction(data[3])
        softmax = bool(int(data[4]))
        weights = FullyConnectedLayer.initializeWeights(size, inputSize, path)
        bias = np.load(path + 'b.npy')
        return FullyConnectedLayer(size, inputSize, activationFunction, \
            weights, bias, softmax)

    def __str__(self):
        return 'Fully Connected Layer\n size: ' + str(self.size) +'\n activation:' + \
                self.activationFunction.getName()

layerMap = {'CONV': ConvolutionLayer, 'MAXP': MaxpoolLayer, \
        'FLAT': FlatteningLayer, 'FCON': FullyConnectedLayer}

def getLayerById(id):
    return layerMap[id]

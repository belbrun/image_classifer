import numpy as np



class Layer:

    def __init__(self, weights, activationFunction):
        self.weights = weights
        self.activationFunction = activationFunction

    def propagateForward(self):
        pass

    def propagateBackwards(self):
        pass


class ConvolutionLayer(Layer):


    def __init__(self, filterNumber, filterSize, stride, activationFunction,\
                 learningRate, inputDepth):

        self.filters = self.initializeFilters(filterNumber, inputDepth, filterSize)
        self.bias = np.zeros(filterNumber)
        self.stride = stride
        self.filterSize = filterSize
        self.learningRate = learningRate
        self.activationFunction = activationFunction
        self.data = None
        self.inputSize = 0

    def initializeFilters(self, filterNumber, inputDepth, filterSize):

        filters = []
        for j in range(0, filterNumber):
            filterGroup = []
            for i in range(0, inputDepth):
                filterGroup.append(np.matrix(np.ones((filterSize,filterSize))))
            filters.append(filterGroup)

        return filters




    def propagateForward(self, input):
        self.data = input
        self.inputSize = input[0].shape[0]
        self.z = []
        outputShape = self.inputSize - self.filterSize + 1
        for (index,filterGroup) in enumerate(self.filters):

            self.z.append(np.matrix(np.zeros((outputShape,outputShape))))

            for i in range(0, self.inputSize - self.filterSize):
                for j in range(0, self.inputSize - self.filterSize):
                    for a in range(0, self.filterSize):
                        for b in range(0, self.filterSize):
                            for (d,x) in enumerate(input):

                                self.z[index][i,j] += filterGroup[d][a,b] * \
                                    x[i*self.stride + a, j*self.stride + b]

                    self.z[index][i,j] += self.bias[index]

        return self.activationFunction.activate(self.z)


    def propagateBackwards(self, errors):

        weightErrors = [[]] * self.filterNumber
        previousErrors = []
        biasError = [0]* self.filterNumber

        for (d,x) in enumerate(self.data):

            previousErrors.append\
                (np.matrix(np.array(self.inputSize,self.inputSize)))

            for (index, filterGroup) in enumerate(self.filters):

                weightErrors[index].append\
                    (np.matrix(np.zeros(self.filterSize,self.filterSize)))

                for i in range(0, self.inputSize - self.filterSize):
                    for j in range(0, self.inputSize - self.filterSize):

                        outputError = errors[index][i,j] * \
                            self.activationFunction.derived(self.z[index][i,j])

                        for a in range(0, self.filterSize):
                            for b in range(0, self.filterSize):

                                weightErrors[index][d][a,b] +=  \
                                    x[i+a, j+b] * outputError
                                previousErrors[d][i*self.stride +a, j*self.stride+b]\
                                    += outputError * self.filters[index][d][a,b]

                    biasErrors[index] += self.bias[index]*outputError

        self.correctWeights(weightErrors, biasErrors)

        return previousErrors

    def correctWeights(weightErrors, biasErrors):

        for (index, filterGroup) in self.filters:
            for (d, filter) in filterGroup:
                for a in range(0, self.filterSize):
                    for b in range(0, self.filterSize):

                        filter[index][d][a,b] += \
                            self.learningRate * weightErrors[index][d][a,b]

            self.bias[index] += self.learningRate* biasErrors[index]



class MaxpoolLayer(Layer):

    def __init__(self, activationFunction, clusterSize = 2):
        self.clusterSize = clusterSize
        self.maxPositions = []
        self.activationFunction = activationFunction
        self.z = []
        self.dataLength = 0


    def propagateForward(self, input):

        self.dataLength = len(input)
        self.inputSize = input[0].shape[0]
        outputShape = self.inputSize - self.clusterSize + 1

        for (d, data) in enumerate(input):

            self.z.append(np.matrix(np.empty((outputShape,outputShape))))
            self.maxPositions.append([])

            for i in range(0, self.inputSize - self.clusterSize + 1 ):
                for j in range(0, self.inputSize - self.clusterSize + 1):

                    max = 0

                    for k in range(0, self.clusterSize):
                        for l in range(0, self.clusterSize):

                            if(data[i+k, j+l] > max):
                                max = data[i+k, j+l]

                    self.maxPositions[d].append((i+k, j+l))
                    self.z[d][i, j] = max

        return self.activationFunction.activate(self.z)


    def propagateBackwards(self, errors):

        previousErrors = []
        outputShape = self.inputSize - self.clusterSize + 1

        for d in range(0, self.dataLength):

            previousErrors.append\
                (np.matrix(np.zeros(self.inputSize,self.inputSize)))

            for i in range(0, outputShape):
                for j in range(0, outputShape):

                    x, y = self.maxPositions[d]
                    previousErrors[d][x,y] = errors[d][i,j]

        return previousErrors

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

        return np.matrix(np.concatenate(input).ravel())

    def propagateBackwards(self, errors):

        previousErrors = []
        layerLength = len(errors)/self.inputDepth


        for layerIndex in range(0, self.inputDepth):

            layerErrors = \
                errors[layerIndex * layerLength: (layerIndex + 1) * layerLength]
            previousErrors.append\
                (layerErrors.resize((self.inputSize, self.inputSize)))

        return previousErrors






class FullyConnectedLayer(Layer):

    def __init__(self, size, inputSize, activationFunction, learningRate):
        self.size = size
        self.inputSize = inputSize
        self.weights = self.initializeWeights(size, inputSize)
        self.bias = [1]*size
        self.activationFunction = activationFunction
        self.learningRate = learningRate
        self.z = None
        self.data = None

    def initializeWeights(self, size, inputSize):

        weights = []
        for i in range(0, size):
            weights.append(np.matrix(np.ones((inputSize,1))))

        return weights

    def propagateForward(self, input):

        self.data = input
        self.z = np.matrix(np.empty((self.size)))
        print(self.z.shape, len(self.weights), len(self.bias))
        for i in range(0, self.size):
            self.z[0,i] = input.dot(self.weights[i]) + self.bias[i]

        return self.activationFunction.activate([self.z])


    def propagateBackwards(self, errors):

        weightErrors = []
        previousErrors = np.matrix(np.zeros(self.inputSize))
        biasErrors = [0] * self.size

        for neuronIndex in range(0, self.size):

            weightErrors.append(np.matrix(np.zeros(self.inputSize)))
            outputError = errors[neuronIndex] * \
                self.activationFunction.derived(self.z[neuronIndex])

            for i in range(self.inputSize):

                weightErrors[neuronIndex][i] += \
                    self.data[i] * outputError
                previousErrors[dataLayer][i] += \
                    self.weights[neuronIndex][i] * outputError

            biasErrors += self.bias[neuronIndex] * outputError

        self.correnctWeights(weightErrors, biasErrors)
        return previousErrors

    def correctWeights(self, weightErrors, biasErrors):

            for neuronIndex in range(0, self.size):
                for i in range(0, self.inputSize):

                    self.weights[neuronIndex][i] += \
                        self.learningRate * self.weightErrors[neuronIndex][i]

                self.bias[neuronIndex] += \
                    self.learningRate * self.biasErrors[neuronIndex]

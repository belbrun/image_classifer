import numpy as np


@abstract
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
        self.learningRate = learningRate
        self.activationFunction = activationFunction
        self.data = None
        self.inputSize = 0

    def initializeFilters(self, filterNumber, inputDepth, filterSize):

        filters = []
        for j in range(0, filterNumber):
            for i in range(0, inputDepth):

                filterGroup = []
                filterGroup.append(np.matrix(np.ones((filterSize,filterSize))))

            filters.append(filterGroup)

        return filters




    def propagateForward(self, input):
        self.data = input
        self.inputSize = input[0].shape[0]
        self.z = []
        outputShape = self.inputSize - self.filterSize + 1
        for (index,filterGroup) in enumerate(self.filters):

            self.z.append(np.matrix(np.zeros(outputShape,outputShape)))

            for i in range(0, self.inputSize - self.filterSize):
                for j in range(0, self.inputSize - self.filterSize):
                    for a in range(0, self.filterSize):
                        for b in range(0, self.filterSize):
                            for (d,x) in enumerate(input):

                                self.z[index][i,j] += self.filterGroup[d][a,b] * \
                                    x[i*self.stride + a, j*self.stride + b]

                    self.z[i,j] += self.bias[index]

            output.append()


        return self.activationFunction.activate(self.z)


    def propagateBackwards(self, errors):
        weightErrors = []
        previousErrors = []
        biasError = []


        for (index, filterGroup) in enumerate(self.filters):

            weightErrors.append([])

            biasErrors.append(0)

            for (d,x) in enumerate(self.data):

                weightErrors[index].append\
                    (np.matrix(np.zeros(self.filterSize,self.filterSize)))
                previousErrors.append\
                    (np.matrix(np.array(self.inputSize,self.inputSize)))

                for i in range(0, self.inputSize - self.filterSize):
                    for j in range(0, self.inputSize - self.filterSize):

                        outputError = errors[d][i,j] * \
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

                        filter[d][a,b] += \
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

            self.z.append(np.matrix(np.empty(outputShape,outputShape)))
            maxPositions.append([])

            for i in range(0, self.inputSize - self.clusterSize + 1 ):
                for j in range(0, self.inputSize - self.clusterSize + 1):

                    max = 0

                    for k in range(0, self.clusterSize):
                        for l in range(0, self.clusterSize):

                            if(data[i+k, j+l] > max):
                                max = data[i+k, j+l]

                    self.maxPositions[d].append((i+k, j+l))
                    self.z[d][i, j] = max

        return self.activationFunction(self.z)


    def propagateBackwards(self, errors):

        previousErrors = []
        outputShape = self.inputSize - self.clusterSize + 1

        for d in range(0, self.dataLength):

            previousErrors.append\
                (np.matrix(np.zeros(self.inputSize,self.inputSize)))

            for i in range(0, outputShape):
                for j in range(0, outputShape):

                    x, y = self.maxPositions[d][0], self.maxPositions[d][1]
                    previousErrors[d][x,y] = errors[d][i,j]

        return previousErrors




class FullyConnectedLayer(Layer):

    def __init__(self, size, inputSize, inputDepth, activationFunction):
        self.size = size
        self.weights = self.initializeWeights(size, inputSize, inputDepth)
        self.bias = [1]*size
        self.activationFunction = activationFunction
        self.z = None
        self.data = None

    def initializeWeights(self, size, inputSize, inputDepth):

        weights = []
        for i in range(0, size):

            weights.append(np.matrix(np.ones(inputSize^2 * inputDepth)))

        return weights

    def propagateForward(self, input):

        self.data = input
        self.z = np.matrix(np.empty(self.size))

        if len(input.shape) > 1:
            input = self.flatten(input)

        for i in range(0, self.size):
            self.z[i] = input.dot(self.weights[i]) + self.bias[i]

        return self.activationFunction.activate([self.z])


    def propagateBackwards(self, errors):

        weightErrors = []
        previousErrors = []
        biasErrors = [0] * self.size
        inputShape = self.data[0].shape

        for i in range(0, len(self.data)):
            previousErrors.append(np.matrix(np.zeros(inputShape)))

        for neuronIndex in range(0, self.size):

            weightErrors.append(np.matrix(np.zeros(inputSize^2 * inputDepth)))

            if len(inputShape) > 1 :

                for i in range(0, inputShape[0]):
                    for j in range(0, inputShape[1]):

                        position = inputShape[0] * i + j
                        outputError = errors[neuronIndex] * \
                            self.activationFunction.derived(self.z[position])

                        for dataLayer in range(0, len(self.data)):

                            weightErrors[neuronIndex][position] += \
                                self.data[dataLayer][i, j] * outputError
                            previousErrors[dataLayer][i, j] += \
                                self.weights[neuronIndex][position] * outputError

                        biasErrors += self.bias[neuronIndex] * outputError


            else:

                for i in range(inputShape[0]):

                    outputError = errors[neuronIndex] * \
                        self.activationFunction.derived(self.z[i])

                    for dataLayer in range(0, len(self.data)):

                        weightErrors[neuronIndex][i] += \
                            self.data[dataLayer][i] * outputError
                        previousErrors[dataLayer][i] += \
                            self.weights[neuronIndex][i] * outputError

                    biasErrors += self.bias[neuronIndex] * outputError

        self.correnctWeights(weightErrors, biasErrors)
        return previousErrors


    def flatten(self, data):

        flatData = np.matrix(np.empty(0))

        for layer in data:
            flatData.append(layer.flatten())

        return flatData

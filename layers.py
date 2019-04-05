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


    def __init__(self, filterNumber, filterSize, stride, activationFunction\
                 learningRate, inputDepth):

        self.filters = initializeFilters(filterNumber, inputDepth, filterSize)
        self.bias = np.zeros(filterNumber)
        self.stride = stride
        self.learningRate = learningRate
        self.activationFunction = activationFunction
        self.data = None
        self.inputSize = 0

    def initializeFilters(filterNumber, inputDepth, filterSize):

        filters = []
        for j in range(0, filterNumber):
            for i in range(0, inputDepth):

                filterGroup = []
                filterGroup.append(np.matrix(np.ones((filterSize,filterSize))))

            filters.append(filterGroup)

        return filters


    @static
    def generateConvolutionLayer():
    """
        Returns a new convolution layer given the input data parameters.
    """

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
                                    x[i*self.stride + a, j*self.stride + b])

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

                                previousErrors[d][i*self.strid +a, j*self.stride+b]\
                                    += outputError * self.filters[index][d][a,b]

                    biasErrors += self.bias[index]*outputError

        self.correctWeights(weightErrors, biasErrors)

        return previousErrors

    def correctWeights(weightErrors, biasError):

        for (index, filterGroup) in self.filters:
            for (d, filter) in filterGroup:
                for a in range(0, self.filterSize):
                    for b in range(0, self.filterSize):

                        filter[d][a,b] += \
                        self.learningRate * weightErrors[index][d][a,b]

            self.bias[index] += self.learningRate* biasError[index]



class MaxpoolLayer(Layer):

    def __init__(self, activationFunction, clusterSize = 2):
        self.clusterSize = clusterSize
        self.maxPositions = []
        self.activationFunction = activationFunction
        self.z = []
        self.dataLength = 0


    def propagateForward(self, input):

        self.dataLength = length(input)
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

        for d in range(0, self.dataLength)

            previousErrors.append\
                (np.matrix(np.zeros(self.inputSize,self.inputSize)))

            for i in range(0, outputShape):
                for j in range(0, outputShape):

                    previousErrors[d]\
                    [self.maxPositions[d][0], self.maxPositions[d][1]] = \
                            errors[d][i,j]

        return previousErrors

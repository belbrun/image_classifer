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
                 learningRate):
        self.filters = []
        self.bias = 0 #TODO: implement multiple filters per layer?
        for i in range(0, filterNumber):
            self.filters.append(np.matrix(np.array((filterSize,filterSize))))
        self.stride = stride
        self.learningRate = learningRate
        self.activationFunction = activationFunction
        self.data = None
        self.inputSize = 0
        self.z = None


    @static
    def generateConvolutionLayer():
    """
        Returns a new convolution layer given the input data parameters.
    """

    def propagateForward(self, input):
        self.data = input
        self.inputSize = input[0].shape[0]
        outputShape = self.inputSize - self.filterSize + 1
        self.z = np.matrix(np.array(outputShape,outputShape))

        for i in range(0, self.inputSize - self.filterSize):
            for j in range(0, self.inputSize - self.filterSize):
                for a in range(0, self.filterSize):
                    for b in range(0, self.filterSize):
                        for (d,x) in enumerate(input):

                            self.z[i,j] += self.filters[d][a,b] * \
                            x[i*self.stride + a, j*self.stride + b])

                self.z[i,j] += self.bias

        return self.activationFunction.activate(self.z)


    def propagateBackwards(self, errors):
        weightErrors = []
        previousErrors = []

        for (d,x) in enumerate(self.data):

            weightErrors.append\ #TODO: check line breaking
            (np.matrix(np.array(self.filterSize,self.filterSize)))
            previousErrors.append\
            (np.matrix(np.array(self.inputSize,self.inputSize)))

            for i in range(0, self.inputSize - self.filterSize):
                for j in range(0, self.inputSize - self.filterSize):

                    outputError = errors[d][i,j] * \
                                self.activationFunction.derived(self.z[i,j])

                    for a in range(0, self.filterSize):
                        for b in range(0, self.filterSize):

                            weightError[d][a,b] +=  \
                            self.data[i+a, j+b] * outputError

                            previousErrors[d][i*self.stride + a, j*self.stride + b] +=\
                            outputError * self.filters[d][a,b]

                    biasError += self.bias*outputError

        self.correctWeights(weightErrors, biasError)

        return previousErrors

    def correctWeights(weightErrors, biasError):

        for filter in self.filters:
            for a in range(0, self.filterSize):
                for b in range(0, self.filterSize):

                    filter[d][a,b] += self.learningRate * weightErrors[d][a,b]

        self.bias += self.learningRate* biasError



class PoolLayer(Layer):

    def __init__(self, activationFunction, clusterSize = 2):
        self.clusterSize = clusterSize
        self.maxPositions = []
        self.activationFunction = activationFunction


    def propagateForward(self, input):

        self.inputSize = input[0].shape[0]
        outputShape = self.inputSize - self.clusterSize + 1
        self.z = np.matrix(np.array(outputShape,outputShape))

        for i in range(0, input.shape[0] - self.clusterSize + 1 ):
            for j in range(0, input.shape[1] - self.clusterSize + 1):

                max = 0

                for k in range(0, self.clusterSize):
                    for l in range(0, self.clusterSize):

                        if(input[i+k, j+l] > max):
                            max = input[i+k, j+l]

                self.maxPositions.append((i,j))
                self.z[i,j] = max

        return self.activationFunction(self.z)


    def propagateBackwards(self):
        pass

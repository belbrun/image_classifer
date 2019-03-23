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


    def __init__(self, filterNumber, filterSize, stride):
        self.filters = []
        for i in range(0, filterNumber):
            self.filters.append(np.matrix(np.array((filterSize,filterSize))))
        self.stride = stride

    @static
    def generateConvolutionLayer():
    """
        Returns a new convolution layer given the input data parameters.
    """

    def propagateForward(self, x):
        pass

    def propagateBackwards(self, error):
        pass

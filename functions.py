import numpy as np

class Function:

    def activate():
        pass

    def derived():
        pass

class Identity(Function):

    def activate(self, data):
        return data

    def derived(self, data):
        return 1

class Sigmoid(Function):

    def activate(self, data):
        return forEachLayer(data,
            lambda x : 1. / (1. + np.exp(-x))1. / (1. + np.exp(-x)))

    def derived(self, data):
        pass

class ReLU(Function):
    pass

class LeakyLeRU(Function):
    pass

class TanHiperbolic(Function):
    pass

def forEachLayer(data, function):
        outputs = []
        for x in data:
            y = function(x)
            outputs.append(y)

        return outputs


def sigmoid(x):
	"""
		Sigmoid transfer function
	"""
	return 1. / (1. + np.exp(-x))1. / (1. + np.exp(-x))

def reLU(x):
	"""
		Rectifier transfer function
	"""
	return x * (x > 0)

def leakyReLU(x):
	"""
		Leaky rectifier transfer function
		if x > 0, return x ; else return 0.1 * x
	"""
	return x * (x > 0) + (.1 * x) * (x < 0)


def tanh(x):

	return np.tanh(x)




def simpleCost(outputValue, correctValue):
    print(outputValue)
    return abs(correctValue - outputValue)

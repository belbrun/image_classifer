from datautil import *
from layers import *
from functions import *
from network import NeuralNetwork
import numpy as np






def main():

    

    testData = []
    testData.append(np.resize(np.arange(9),(3,3)))
    testData.append(np.resize(np.arange(9,18),(3,3)))
    testData.append(np.ones((3,3)))
    print(testData)
    layer = ExtremumPoolLayer(2, 'min')
    print(layer.propagateForward(testData))
    testErrors = []
    testErrors.append(np.resize(np.arange(4),(2,2)))
    testErrors.append(np.resize(np.arange(4,8),(2,2)))
    testErrors.append(np.ones((2,2)))

    print(layer.propagateBackwards(testErrors, 1))

    return








if __name__ == '__main__':
    main()

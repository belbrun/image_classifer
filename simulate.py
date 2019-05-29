from datautil import *
from layers import *
from functions import *
from network import NeuralNetwork
import numpy as np






def main():
    #rgbMatrices =getInput('Im126_1.tif')
    net = NeuralNetwork.load('network_data/old_network')
    datautil.getImageFromArrays(net.layers[0].propagateForward(datautil.getInput('Im001_1.tif')))
    return
    testData = []
    testData.append(np.resize(np.arange(9),(3,3)))
    testData.append(np.resize(np.arange(9,18),(3,3)))
    testData.append(np.ones((3,3)))
    print(testData)
    print(ExtremumPoolLayer(2).propagateForward(testData))
    return


    neuralNet = NeuralNetwork()
    neuralNet.addLayer(ConvolutionLayer(3,3,1,ReLU(),0.1,3))
    neuralNet.addLayer(ExtremumPoolLayer(3))
    neuralNet.addLayer(FlatteningLayer())
    neuralNet.addLayer(FullyConnectedLayer(3, 193548, TanHiperbolic(),0.1))
    neuralNet.save('network_data/')
    #print(neuralNet.feed([rgbMatrices]))
    #neuralNet.train([rgbMatrices],[[4000,2222,3333]], 0.1)





if __name__ == '__main__':
    main()

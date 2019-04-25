from dataloader import *
from layers import *
from functions import *
from network import NeuralNetwork
import numpy as np






def main():
    #rgbMatrices =getInput('Im126_1.tif')
    testData = []
    testData.append(np.resize(np.arange(9),(3,3)))
    testData.append(np.resize(np.arange(9,18),(3,3)))
    layer = ConvolutionLayer(3,3,1,ReLU(),0.1,3)
    output = layer.propagateForward(testData)
    errors = layer.propagateBackwards(output)
    print(testData, output, errors)
    return

    print(rgbMatrices[0].shape)
    neuralNet = NeuralNetwork()
    neuralNet.addLayer(ConvolutionLayer(3,3,1,Identity(),0.1,3))
    neuralNet.addLayer(MaxpoolLayer(Identity()))
    neuralNet.addLayer(FlatteningLayer())
    neuralNet.addLayer(FullyConnectedLayer(3, 193548, Identity(),0.1))
    #print(neuralNet.feed([rgbMatrices]))
    neuralNet.train([rgbMatrices],[[4000,2222,3333]], 0.1)





if __name__ == '__main__':
    main()

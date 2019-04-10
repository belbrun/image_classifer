from dataloader import getInput
from layers import *
from functions import Identity
from network import NeuralNetwork

def main():
    rgbMatrices =getInput('Im126_1.tif')
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

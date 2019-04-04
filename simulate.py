from dataloader import getInput
from layers import *
from functions import Identity

def main():
    rgbMatrices =getInput('Im126_1.tif')
    print(rgbMatrices[0].shape)
    layer = ConvolutionLayer(3, 3, 1, Identity())
    output = layer.propagateForward(rgbMatrices)
    print(output.shape)
    layer = MaxpoolLayer(Identity(),7)
    output = layer.propagateForward(output)
    print(output.shape)



if __name__ == '__main__':
    main()

from dataloader import getInput
from layers import ConvolutionLayer
from functions import Identity

def main():
    rgbMatrices =getInput('Im126_1.tif')
    print(rgbMatrices)

    layer = ConvolutionLayer(3, 3, 1, Identity())
    print(layer.propagateForward(rgbMatrices))

if __name__ == '__main__':
    main()

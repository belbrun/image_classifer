from dataloader import getInput
import layers
import functions

def main():
    rgbMatrices =getInput('Im126_1.tif')
    print(rgbMatrices)

    layer = ConvolutionLayer(3, 3, 1, Identity(), 0.1)
    print(layer.propagateForward(rgbMatrices))

if __name__ == '__main__':
    main()

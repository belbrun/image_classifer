import numpy as np
from PIL import Image

def avaragePool(pixelArrays, clusterSize = 2, stride = 2):

    inputShape = pixelArrays[0].shape
    outputShape = (int((inputShape[0] - clusterSize)/stride) + 1,\
                int((inputShape[1] - clusterSize)/stride) + 1)
    output = []

    for (index,pixelArray) in enumerate(pixelArrays):

        output.append(np.empty(outputShape))

        for i in range(0, outputShape[0]):
            for j in range(0, outputShape[1]):

                sum = 0

                for a in range(0, clusterSize):
                    for b in range(0, clusterSize):

                        sum += pixelArray[i * stride + a,j * stride + b]

                output[index][i,j] = sum/(clusterSize * clusterSize)

    return output

def toGrayScale(image):
    return image.convert('L')


def crop(image, shape):

    height, width = image.size
    cropHeight, cropWidth = int((height-shape[0])/2), int((height-shape[0])/2)
    return image.crop((cropHeight, cropWidth, height-cropHeight, width-cropWidth))
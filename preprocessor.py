import numpy as np
from PIL import Image

def splitToComponents(pixelArray):
    output = []
    rows,columns = pixelArray.shape[0], pixelArray.shape[1]
    redArray = np.empty((rows,columns))
    greenArray = np.empty((rows,columns))
    blueArray = np.empty((rows,columns))

    for i in range(0, columns-1):
        for j in range(0, rows-1):
            redArray[i,j] = pixelArray[i,j,0]
            greenArray[i,j] = pixelArray[i,j,1]
            blueArray[i,j] = pixelArray[i,j,2]

    return [redArray, greenArray, blueArray]

def processArrays(pixelArrays, gray = False, avaraged = False):
    output = []

    if gray:
        output.append(pixelArrays)
    else:
        output.extend(splitToComponents(pixelArrays))

    if avaraged:
        output = avaragePool(output)

    return normalize(output)


def getImageAsArrays(name, path, gray = False, shape = None, rotations = False):
    image = Image.open(path + name, 'r')
    imageAsArrays = []

    if rotations:
        images = getRotations(image)
        for image in images:
            imageAsArrays.append(np.array(processImage(image, gray, shape)))
    else :
        imageAsArrays.append(np.array(processImage(image, gray, shape)))

    return imageAsArrays

def  getImageFromArrays(arrays, gray = False):

    imageArray = None
    type = None

    if gray :
        type = 'L'
        imageArray = arrays[0]*255
    else :
        type = 'RGB'
        redArray,greenArray,blueArray = arrays[0]*255,arrays[1]*255,arrays[2]*255
        imageArray = np.empty((redArray.shape[0], redArray.shape[1], 3))
        for i in range(0, redArray.shape[0]):
            for j in range(0, redArray.shape[1]):
                imageArray[i,j] = \
                    np.array([redArray[i,j], greenArray[i,j], blueArray[i,j]])
    image = Image.fromarray(imageArray.astype('uint8'), type)
    return image



def processImage(image, gray, shape):
    if gray:
        image = toGrayScale(image)
    if shape:
        image = crop(image, shape)
    return image

def saveData(path, data):
    with open(path + 'config.txt', 'w+') as file:
        for information in data:
            file.write(str(information)+'|')


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

def normalize(images):
    normalizedImages = []
    for image in images:
        #print('Image: ', image)
        max = np.max(image)
        min = np.min(image)
        #print('MAXMIN: ', max, min)

        normalizedImages.append((image - min)/(max-min))
        #print('NORML', normalizedImages)
    return normalizedImages


def toGrayScale(image):
    return image.convert('L')


def crop(image, shape):

    height, width = image.size
    cropHeight, cropWidth = (height-shape[0])//2, (width-shape[1])//2
    return image.crop((cropHeight, cropWidth, height-cropHeight, width-cropWidth))

def getRotations(image):
    outputs = []
    for i in range(0, 360, 90):
        outputs.append(image.rotate(i))
    return outputs

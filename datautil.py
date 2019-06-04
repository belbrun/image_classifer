from PIL import Image
import preprocessor as pp
import numpy as np
import os

#def loadImages

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


def getInput(name, path = 'dataset/2/ALL_IDB2/img/', gray = False, shape = None,\
    rotations = False, avaraged = False):

    imageAsArrays = getImageAsArrays(name, path, gray, shape, rotations)
    output = []

    for rotation in imageAsArrays:
        output.append(processArrays(rotation, gray, avaraged))

    return output

def processArrays(pixelArrays, gray = False, avaraged = False):
    output = []

    if gray:
        output.append(pixelArrays)
    else:
        output.extend(splitToComponents(pixelArrays))

    if avaraged:
        output = pp.avaragePool(output)

    return normalize(output)


def getImageAsArrays(name, path, gray = False, shape = None, rotations = False):
    image = Image.open(path + name, 'r')
    imageAsArrays = []

    if rotations:
        images = pp.getRotations(image)
        for image in images:
            imageAsArrays.append(np.array(processImage(image, gray, shape)))
    else :
        imageAsArrays.append(np.array(processImage(image, gray, shape)))

    return imageAsArrays

def processImage(image, gray, shape):
    if gray:
        image = pp.toGrayScale(image)
    if shape:
        image = pp.crop(image, shape)
    return image

def saveData(path, data):
    with open(path + 'config.txt', 'w+') as file:
        for information in data:
            file.write(str(information)+'|')

def loadData(path):
    with open(path + 'config.txt', 'r+') as file:
        return file.read().split('|')

def makeDirectory(path):
    os.mkdir(path)

def writeLog(path, log):
    with open(path + 'log.txt', 'w+') as file:
        for line in log:
            file.write(line + '\n')

def getLayerIds(path):
    layerIds = []
    numOfLayers = len([i for i in os.listdir(path) if not i.endswith('.txt')])
    for i in range(0, numOfLayers):
        with open(path + str(i) + '/config.txt', 'r+') as file:
            layerIds.append(file.read(4))
    return layerIds

def getImageFromArrays(arrays, gray = False):

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
    image.show()



if __name__ == '__main__':
    arrays = getInput('Im221_0.tif', gray = True, shape = (200,200), avaraged = True, rotations = False)
    print(arrays[0][0].shape)
    getImageFromArrays(arrays[2], True)

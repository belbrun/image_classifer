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

def getInput(name, path = 'dataset/2/ALL_IDB2/img/', gray = False, shape = None,\
    avaraged = False, clusterSize = 2, stride = 2):


    pixelArray = getImageAsVector(name, path, gray, shape)
    output = []

    if gray:
        output.append(pixelArray)
    else:
        rows,columns = pixelArray.shape[0], pixelArray.shape[1]
        redArray = np.empty((rows,columns))
        greenArray = np.empty((rows,columns))
        blueArray = np.empty((rows,columns))

        for i in range(0, columns-1):
            for j in range(0, rows-1):
                redArray[i,j] = pixelArray[i,j,0]
                greenArray[i,j] = pixelArray[i,j,1]
                blueArray[i,j] = pixelArray[i,j,2]
        output.append(redArray)
        output.append(greenArray)
        output.append(blueArray)

    if avaraged:
        output = pp.avaragePool(output)

    return normalize(output)

def getImageAsVector(name, path, gray = False, shape = None):
    image = Image.open(path + name, 'r')
    if gray or shape:
        image = processImage(image, gray, shape)
    pixelArray = np.array(image)
    return pixelArray

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
    dirList = os.listdir(path)
    for i in range(0, len(dirList) - 1):
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
    print(image.size)
    image.show()



if __name__ == '__main__':
    arrays = getInput('Im200_0.tif', gray = True,shape =(230,230), avaraged = True)
    getImageFromArrays(arrays, True)

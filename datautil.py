from PIL import Image
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

def getInput(name, path = 'dataset/2/ALL_IDB2/img/' ):

    pixelArray = getImageAsVector(name, path)
    rows,columns = pixelArray.shape[0], pixelArray.shape[1]
    redArray = np.empty((rows,columns))
    greenArray = np.empty((rows,columns))
    blueArray = np.empty((rows,columns))

    for i in range(0, columns-1):
        for j in range(0, rows-1):

            redArray[i,j] = pixelArray[i,j,0]
            greenArray[i,j] = pixelArray[i,j,1]
            blueArray[i,j] = pixelArray[i,j,2]

    return normalize([redArray, greenArray, blueArray])

def getImageAsVector(name, path):
    image = Image.open(path + name, 'r')
    image = image.crop((30,30,227,227))
    #image = image.resize((20,180), Image.BILINEAR)
    #image.show()
    pixelArray = np.array(image)
    return pixelArray


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

if __name__ == '__main__':
    print(len(getInput('Im001_1.tif')))
    print(len(getInput('Im131_0.tif')))

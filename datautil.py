from PIL import Image
import preprocessor as pp
import numpy as np
import os
import parser

#def loadImages

#TODO: refactor datautil module to interact only with data
#add rest of the logic to util, preprocessor, a possible parser




def getInput(name, path = 'dataset/2/ALL_IDB2/img/', gray = False, shape = None,\
    rotations = False, avaraged = False):

    imageAsArrays = pp.getImageAsArrays(name, path, gray, shape, rotations)
    output = []

    for rotation in imageAsArrays:
        output.append(pp.processArrays(rotation, gray, avaraged))

    return output



def loadData(path, old = False):
    name = 'old_config.txt' if old else 'config.txt'
    with open(path + name, 'r+') as file:
        return file.read().split('|')

def makeDirectory(path):
    os.mkdir(path)

def writeLog(path, log):
    with open(path + 'log.txt', 'w+') as file:
        for line in log:
            file.write(line + '\n')

def readLog(path):
    with open(path + 'log.txt', 'r+') as file:
        return file.read().split('\n')

def writeResults(path, results):
    with open(path + 'results.txt', 'w+') as file:
        for line in results:
            file.write(str(line) + '|')

def readResults(path):
    with open(path + 'results.txt', 'r+') as file:
        return file.read().split('|')

def getLayerIds(path):
    layerIds = []
    numOfLayers = len([i for i in os.listdir(path) if not i.endswith('.txt')])
    for i in range(0, numOfLayers):
        with open(path + str(i) + '/config.txt', 'r+') as file:
            layerIds.append(file.read(4))
    return layerIds

def getValidationResults(path):
    log = readLog(path)
    return parser.parseLinesForResults(log, 'VALIDATION RESULTS')

def getTrainingResults(path):
    log = readLog(path)
    return parser.parseLinesForResults(log, '[')

def getTestResults(path):
    resultsString = readResults(path)
    return  parser.parseLinesForResults(resultsString)

def getEpochValidationOutputs(epochPath, epoch):
    return parser.parseEpochValidationOutputs(readLog(epochPath), epoch)


if __name__ == '__main__':
    arrays = getInput('Im001_1.tif', gray = True, shape = (200,200), avaraged = True, rotations = True)
    #arrays = getInput('Im001_1.tif', gray = True)
    #arrays = getInput('Im001_1.tif', shape = (200,200))
    #arrays = getInput('Im001_1.tif', rotations = True)
    #arrays = getInput('Im001_1.tif', avaraged = True)
    #arrays = getInput('Im001_1.tif')

    for i, array in enumerate(arrays):

        image = getImageFromArrays(array, True)
        image.save('dataset/examples/' + str(i) + '.png', 'png')

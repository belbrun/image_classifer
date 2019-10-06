from PIL import Image
import preprocessor as pp
import numpy as np
import os
import parser
import struct as st
import preprocessor

#def loadImages

#TODO: refactor datautil module to interact only with data
#add rest of the logic to util, preprocessor, a possible parser

def getLabel(file, index):
    file.seek(0)
    magic = st.unpack('>4B',file.read(4)) #  magic number
    imgNum = st.unpack('>I',file.read(4))[0] #  num of images
    file.seek(index, 1)
    return st.unpack('>'+'B',file.read(1))[0]

def getImage(file, index):
    file.seek(0)
    magic = st.unpack('>4B',file.read(4)) #  magic number
    # read dimensions
    imgNum = st.unpack('>I',file.read(4))[0] # num of images
    rowNum = st.unpack('>I',file.read(4))[0] # num of rows
    columnNum = st.unpack('>I',file.read(4))[0] # num of column

    size = rowNum * columnNum
    file.seek(index*size, 1)

    return np.asarray(st.unpack('>'+'B'*size,file.read(size)))\
            .reshape((rowNum, columnNum))

def getEntity(index, test = False):
    path = 'mnist/'
    image = 't10k-images-idx3-ubyte' if test else 'train-images-idx3-ubyte'
    label = image.replace('images-idx3', 'labels-idx1')
    return (getImage(open(path+image, 'rb'), index), \
            getLabel(open(path+label, 'rb'), index))



def saveData(path, data):
    """
        Save data to a given path in config.txt file.
    """
    with open(path + 'config.txt', 'w+') as file:
        for information in data:
            file.write(str(information)+'|')


def loadData(path, old = False):
    """
        Load the data from a given path, from config.txt file.
    """
    name = 'old_config.txt' if old else 'config.txt'
    with open(path + name, 'r+') as file:
        return file.read().split('|')

if __name__ == '__main__':
    im = getEntity(1271, True)
    print(im[1])
    preprocessor.getImageFromArrays([im[0]], True).show()

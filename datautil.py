from PIL import Image
import preprocessor as pp
import numpy as np
import os
import parser

#def loadImages

#TODO: refactor datautil module to interact only with data
#add rest of the logic to util, preprocessor, a possible parser




def saveData(path, data):
    """
        Save data to a given path in config.txt file.
    """
    with open(path + 'config.txt', 'w+') as file:
        for information in data:
            file.write(str(information)+'|')


def loadData(path, old = False):
    """
        Load the data from a fiven path, from config.txt file.
    """
    name = 'old_config.txt' if old else 'config.txt'
    with open(path + name, 'r+') as file:
        return file.read().split('|')

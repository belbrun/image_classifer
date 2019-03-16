from PIL import Image
import numpy as np

#def loadImages

def getImageAsVector(name, path = 'dataset/2/ALL_IDB2/img/'):
    image = Image.open(path + name, 'r')
    pixelArray = np.asarray(image)
    return pixelArray


def main():
    pixelArray = getImageAsVector('Im126_1.tif')
    print (pixelArray)
    print (pixelArray.shape)

if __name__ == '__main__':
    main()

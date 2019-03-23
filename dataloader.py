from PIL import Image
import numpy as np

#def loadImages

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
    return [np.matrix(redArray),\
        np.matrix(greenArray), np.matrix(blueArray)]

def getImageAsVector(name, path):
    image = Image.open(path + name, 'r')
    pixelArray = np.asarray(image)
    return pixelArray


def main():
    rgbMatrices =getInput('Im126_1.tif')
    print(rgbMatrices[0][255])

if __name__ == '__main__':
    main()

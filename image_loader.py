from PIL import Image
import os
import numpy as np

custom_npys = [
    'QuestionMark.npy',
    'SadFace.npy',
    'Egg.npy'
]

npys = [
    'Circle.npy',
    'SmileyFace.npy',
    'Square.npy',
    'Tree.npy',
    'Triangle.npy',
    'House.npy'
]

def loadImages():
    # The output is training_data, a tuple with an array of 
    # pixels and the desired output for the network.
    training_data = []
    # load custom data
    for i in range(len(custom_npys)):
        y = np.zeros(len(custom_npys) + 7)  #Desired output
        y[i] = 1
        a = np.load(custom_npys[i])
        counter = 0
        for imagen in a:
            if (len(imagen) == 784):
                training_data.append((normalizePixels(imagen, True), y))
    
    # load npys from google cloud project 
    for i in range(len(npys)):
        y = np.zeros(len(npys) + 4)  #Desired output
        y[i + 3] = 1
        a = np.load(npys[i])
        for i in range(10000):
            if (len(a[i]) == 784):
                training_data.append((normalizePixels(a[i], False), y))
    
    #load MickeyMouse from jpegs and bmps
    path = 'data\\MickeyMouse\\'
    y = np.zeros(10)
    y[9] = 1
    for r, d, f in os.walk(path):
        for file in f:
            im = list(Image.open(path + file).getdata())
            if '.bmp' in file:
                if (len(im) == 784):
                    training_data.append((normalizePixels(getJpgPixels(im), True), y))
            else:
                if (len(im) == 784):
                    training_data.append((normalizePixels(im, True), y))
    return training_data

# .bmp has an RGB tuple, needs to be converted
def getJpgPixels(tuples):
    out = []
    for tuple in tuples:
        pixel = sum(tuple) / len(tuple)
        out.append(pixel)
    return out


def normalizePixels(pixels, inverted):
# Receives ints between 0 and 255 and returns 0 or 1
# also flattens the array so its 1x28
    out = []
    if inverted:
        for pixel in pixels:
            if int(pixel) > 250:
                out.append(0) 
            else:
                out.append(1)
    else:
        for pixel in pixels:
            if int(pixel) > 250:
                out.append(1) 
            else:
                out.append(0)
    return np.array(out).reshape(784,1)
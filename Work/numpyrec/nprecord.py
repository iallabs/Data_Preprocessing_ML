import os 
import numpy as np
from glob import glob
import cv2 as cv
###############
# Cat vs Dog #
###############

IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_CLASS = 2
gen_file_path = "train/train/"

def get_paths(ur_path):
    file_class1 = os.path.join(gen_file_path,'cat*.jpg' )
    file_class2 = os.path.join(gen_file_path, 'dog*.jpg')
    cat_files = glob(file_class1)
    dog_file = glob(file_class2)
    size = len(cat_files) + len(dog_file)
    return cat_files, dog_file, size

def get_numpy(class1_files, class2_files, size):
    dataX = np.zeros((size,IMAGE_HEIGHT, IMAGE_WIDTH,IMAGE_CHANNELS), dtype=np.float64)
    dataY = np.zeros((size))
    i = 0
    for file in class1_files:
            img = cv.imread(file)
            img = cv.resize(img,(IMAGE_HEIGHT, IMAGE_WIDTH))
            img_np = np.array(img)
            dataX[i] = img_np
            dataY[i]= 0
            i += 1
    for file in class2_files:
        try:
            img = cv.imread(file)
            img = cv.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
            img_np = np.array(img)
            dataX[i] = img_np
            dataY[i] = 1
            i +=1
        except:
            continue
    return dataX, dataY

def make_npfile(dir, inputs, labels):
    ######
    #Space dedicated for pre-processing methods
    ######
    if  (dir == ""):
        dir = os.getcwd()
    np.save(dir, [inputs, labels], ["inputs, labels"])
    return 0

def load_npfile(filename):
    f = os.path.join(os.path.dirname(filename), filename)
    data = np.load(f)
    features = data[0]
    labels = data[1]
    return features, labels
alpha, beta, gamma = get_paths(gen_file_path)
a , b = get_numpy(alpha, beta, gamma)
make_npfile("", a, b)
'''a, b= load_npfile("train.npy")'''

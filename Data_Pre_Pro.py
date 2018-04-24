import cv2  #"load image Opencv library"
import glob
import zipfile
import numpy as np
import os
from os.path import isfile, join  #manipulation of file paths
from os import listdir #get list of all the files in a directory
from random import shuffle


def dezipper(filezip, pathdst=''):
    if pathdst =='':
        pathdst = os.getcwd()
    zfile = zipfile.ZipFile(filezip, 'r')
    os.makedirs(zfile.filename.split(".")[1])
    for i in zfile.namelist():
        if os.path.isdir(i):
            try: os.makedirs(pathdst + os.sep + i)
            except: pass
        else:
            try: os.makedirs(pathdst + os.sep + os.path.dirname(i))
            except: pass
            data = zfile.read(i)
            with open(pathdst + os.sep + i,"wb") as fp:
                fp.write(data)
                fp.close()
    zfile.close()


def seperateData(data_dir):    for filename in listdir(data_dir):
        if isfile(join(data_dir, filename)):
            tokens = filename.split('.')
            if tokens[-1] == 'jpg':
                image_path = join(data_dir, filename)
                if not os.path.exists(join(data_dir, tokens[0])):
                    os.makedirs(join(data_dir, tokens[0]))
                copyfile(image_path, join(join(data_dir, tokens[0]),filename))
                image.remove(image_path)

class DataSetGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_labels = self.get_data_labels()
        self.data_info = self.get_data_paths()
    
    def get_data_labels(self):
        data_labels = []
        for filename in listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                data_labels.append(filename)
        return data_labels
    
    def get_data_paths(self):
        data_paths = []
        for label in self.data_labels:
            img_lists = []
            path = join(self.data_dir, label)
            for filename in listdir(path):
                tokens = filename.split('.')
                if tokens[-1]=='jpg':
                    image_path = join(path, filename)
                    img_lists.append(image_path)
            shuffle(img_lists)
            data_paths.append(img_lists)
        return data_paths

    def get_mini_batches(self, batch_size=10, image_size=(200, 200), allchannel=True):
        images = []
        labels = []
        empty = False
        count = 0
        single_batch_size = int(batch_size/len(self.data_info))
        while True: 
            for i in range(len(self.data_labels)):
                label = np.zeros(len(self.data_labels), dtype = int)
                label[i] = 1
                if len(self.data_info[i]) < count+1:
                    empty=True
                    continue
                empty = False
                img = cv2.imread(self.data_info[i][count])
                img = self.resizeAndPad(img, image_size)
                if not allchannel:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                images.append(img)
                labels.append(label)
            count+=1

            if empty:
                break
            # if the iterator is multiple of batch size return the mini batch
            if (count)%single_batch_size == 0:
                yield np.array(images,dtype=np.uint8), np.array(labels,dtype=np.uint8)
                del images
                del labels
                images=[]
                labels=[]


    def resizeAndPad(self, img, size):
        h, w = img.shape[:2]

        sh, sw = size
        # interpolation method
        if (h > sh) or (w > sw):  # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h

        # padding
        if aspect > 1: # horizontal image
            new_shape = list(img.shape)
            new_shape[0] = w
            new_shape[1] = w
            new_shape = tuple(new_shape)
            new_img=np.zeros(new_shape, dtype=np.uint8)
            h_offset=int((w-h)/2)
            new_img[h_offset:h_offset+h, :, :] = img.copy()

        elif aspect < 1 : # vertical image
            new_shape = list(img.shape)
            new_shape[0] = h
            new_shape[1] = h
            new_shape = tuple(new_shape)
            new_img = np.zeros(new_shape,dtype=np.uint8)
            w_offset = int((h-w) / 2)
            new_img[:, w_offset:w_offset + w, :] = img.copy()
        else:
            new_img = img.copy()
        # scale and pad
        scaled_img = cv2.resize(new_img, size, interpolation=interp)
        return scaled_img

#This line can be
dezipper("./train_catvs_dog.zip")
seperateData("./train_catvs_dog")
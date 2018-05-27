from matplotlib import pyplot as plt
import cv2
import numpy as np

def plot_histogram(frame, color='b', channels=[0], mask=None, histosize=[256], ranges=[0,256]):
    histr = cv2.calcHist([frame], channels, mask, histosize, ranges)
    plt.plot(histr, color=color)
    plt.show()

def plot_bgr_histogram(frame, mask=None, histosize=[256], ranges=[0, 256]):
    color = ('b','g','r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([frame], [i], mask, [256], ranges)
        plt.plot(histr, color=col)
        plt.xlim([0,256])
    plt.show()

import numpy as np
import cv2
'''from ..colors.colors import thresh_bin'''
import matplotlib.pyplot as plt

####################
# Canny contours
####################
def canny_cont(frame, threch1, threch2):
    img_save = frame
    edges = cv2.Canny(frame, threch1,threch2)
    return edges


###################
# Thresholding
####################
def thresh_bin(frame, value_thr, value_aft):
    _ , img  = cv2.threshold(frame, value_thr, value_aft, cv2.THRESH_BINARY)
    return img

def find_contours(frame, approx=False):
    tresh = thresh_bin(frame, 70, 255)
    if approx:
        image, contours, hierarchy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        image, contours, hierarchy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return image, contours, hierarchy

    #image: source image
    # contours: contour retrieval mode
    # hierarchy: contour approximation method

def draw_all_contours(frame):
    save_img = frame
    tresh, contours, _ = find_contours(frame)
    img = cv2.drawContours(tresh, contours, -1,(0,255))
    return img, save_img

def draw_contour(frame, pos):
    #pos is an int begining from 0
    save_img = frame
    tresh, contours, _ = find_contours(frame)
    image = cv2.drawContours(tresh, contours, pos, (0,255))
    return image, save_img

def draw_contour_approx(frame):
    save_img = frame
    tresh, contours, _ = find_contours(frame, approx=True)
    image = cv2.drawContours(tresh, contours, -1, (0,255))
    return image, save_img

####################
# Contours Features
####################

def get_moment(frame, index):
    _, contours, _ = find_contours(frame, approx=True)
    cnt = contours[index]
    M = cv2.moments(cnt)
    return M          #M is a dict of all the moments


def get_contour_area(contour):
    area = cv2.contourArea(contour)
    return area

def get_perimeter(contour):
    perimeter = cv2.arcLength(contour)
    return perimeter

def get_approx_cont(contour):
    epsilon = 0.1*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    return approx


####################
# Bounding Rectangle
####################
def bounding_rec(frame):
    _, contours,_ = find_contours(frame, approx=True)
    image = frame
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        approx = get_approx_cont(c)
        if len(approx) >= 4:
            image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0))
    
    return image

def draw_rec(frame, index):
    _,contours,_ = find_contours(frame, approx=True)
    cnt = contours
    rect = cv2.minAreaRect(cnt[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    image = cv2.drawContours(frame, [box],0,(0,0,255),2)
    return image


########################
# Contours Properties
########################
def get_ratio(contour):
    x,y,w,h = cv2.boundingRect(contour)
    ratio = float(w)/h
    return ratio

def get_extent(contour):
    area = get_contour_area(contour)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent

def mask_pix(frame_gray):
    _,contours,_ = find_contours(frame_gray)
    mask = np.zeros(frame_gray.shape,np.uint8)
    cv2.drawContours(mask,contours,0,255)
    pixelpoints = cv2.findNonZero(mask)
    return pixelpoints

#Aspect ratio: It is the ratio of width to height of bounding rect of the object


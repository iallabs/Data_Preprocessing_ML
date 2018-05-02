import cv2
import numpy as np
import matplotlib.pyplot as plt
#Let's consider M the translation matrix. Given a translation vector (tx,ty),
# the matrix is defined as follows: M = [1 0 tx]
#                                       [0 1 ty]


##################
# Translation
##################
def translate(frame, tx, ty):
    dim = frame.shape
    M = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(frame, M, dim)
    return img

################
# Rotation
################
#See reference: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
#Angle is in degree
def rotate(frame, angle):
    dim = (int(frame.shape[0]/2), int(frame.shape[1]/2))
    M = cv2.getRotationMatrix2D(dim, angle, 1) #Last argument is for scale
    img = cv2.warpAffine(frame, M, dim)
    return img

#######################
# Affine transformation
#######################
# For this func , we need three points from the original image
# And the corresponding points in the output image 
#list1: list of tuples of input image: 3 tuples// Same for list2 (output image) 
def affine(frame, list1, list2):
    '''assert type(list1) == list and len(list1)==3'''
    col,row,_ = frame.shape
    sub1, sub2 = [], []
    for i in list1:
        sub1.append([i[0],i[1]])
    for j in list2:
        sub2.append([j[0], j[1]])
    ps1 = np.float32(sub1)
    ps2 = np.float32(sub2)
    M = cv2.getAffineTransform(ps1,ps2)
    img = cv2.warpAffine(frame, M, dsize=(col,row))
    return img



#######################
# Perspective transformation
#######################

def perspec(frame, list1, list2):
    '''assert type(list1) == list and len(list1)==4'''
    col,row,_ = frame.shape
    sub1, sub2 = [], []
    for i in list1:
        sub1.append([i[0],i[1]])
    for j in list2:
        sub2.append([j[0], j[1]])
    ps1 = np.float32(sub1)
    ps2 = np.float32(sub2)
    M = cv2.getPerspectiveTransform(ps1,ps2)
    img = cv2.warpPerspective(frame, M, dsize=(col,row))

    return img

'''a = cv2.imread("C:/Users/User12/Desktop/Work/train/train/cat.0.jpg", cv2.IMREAD_UNCHANGED)
b = perspec(a, [(50,50),(300,50),(50,300), (300,300)], [(0,0),(250,0),(0,250),(250,250)])
plt.imshow(b)
plt.show()'''


#######################
# Morphological transformation
#######################
#TODO:



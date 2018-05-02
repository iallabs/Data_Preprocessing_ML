import cv2

#############
#Color change
#############
#Changing color space:
def change_color_gray(frame):
    change = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return change

def change_color_hsv(frame):
    change = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return change

def change_color_rgb(frame):
    change = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return change

#Object tracking: using a specified range of color
#Ex: lower_blue = np.array([110,50,50]); upper_blue = np.array([130,255,255])
#lower_col & upper_col have to be numpy arrays
def track_color(frame, lower_col, upper_col):
    hsv = change_color_hsv(frame)
    mask = cv2.inRange(hsv, lower_col, upper_col)
    img = cv2.bitwise_and(frame,frame,mask=mask)
    return img

#############
#Thresholding
#############

#Images processed in this part must be grayscale
#Ex: if x>value_min --> white else black
# value_aft is given if pixel > value_thr

def thresh_bin(frame, value_thr, value_aft):
    _ , img  = cv2.threshold(frame, value_thr, value_aft, cv2.THRESH_BINARY)
    return img

def thresh_bin_inv(frame, value_thr, value_aft):
    _ , img  = cv2.threshold(frame, value_thr, value_aft, cv2.THRESH_BINARY_INV)
    return img

def thresh_truc(frame, value_thr, value_aft):
    _ , img  = cv2.threshold(frame, value_thr, value_aft, cv2.THRESH_TRUNC)
    return img

def thresh_tozero(frame, value_thr, value_aft):
    _ , img  = cv2.threshold(frame, value_thr, value_aft, cv2.THRESH_TOZERO)
    return img

def thresh_tozero_inv(frame, value_thr, value_aft):
    _ , img  = cv2.threshold(frame, value_thr, value_aft, cv2.THRESH_TOZERO_INV)
    return img

#Adaptive thresholding
#block_h: means block size height, block_w means block size width

##With binary thresholding

def thresh_adapgauss_bin(frame, value_aft, block_h, block_w):
    img = cv2.adaptiveThreshold(frame, value_aft, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY, block_h, block_w )

    return img

def thresh_adapgauss_bininv(frame, value_aft, block_h, block_w):
    img = cv2.adaptiveThreshold(frame, value_aft, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY_INV, block_h, block_w )

    return img

def thresh_adapgauss_tunc(frame, value_aft, block_h, block_w):
    img = cv2.adaptiveThreshold(frame, value_aft, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_TRUNC, block_h, block_w )

    return img

def thresh_adapgauss_toz(frame, value_aft, block_h, block_w):
    img = cv2.adaptiveThreshold(frame, value_aft, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_TOZERO, block_h, block_w )

    return img

def thresh_adapgauss_tozinv(frame, value_aft, block_h, block_w):
    img = cv2.adaptiveThreshold(frame, value_aft, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_TOZERO_INV, block_h, block_w )

    return img

def thresh_adapmean_bin(frame, value_aft, block_h, block_w):
    img = cv2.adaptiveThreshold(frame, value_aft, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY, block_h,block_w)
    return img

def thresh_adapmean_trunc(frame, value_aft, block_h, block_w):
    img = cv2.adaptiveThreshold(frame, value_aft, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_TRUNC, block_w, block_h)
    return img

def thresh_adapmean_toz(frame, value_aft, block_h, block_w):
    img = cv2.adaptiveThreshold(frame, value_aft, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_TOZERO, block_h, block_w)
    return img

def thresh_adapmean_tozinv(frame, value_aft, block_h, block_w):
    img = cv2.adaptiveThreshold(frame, value_aft, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_TOZERO_INV, block_h, block_w)
    return img


#################
#Otsu thresholding
##################

#TODO: ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
import tensorflow as tf
import cv2

def normalize(image, label):
  """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  return image, label

####################
# Image Gradient
####################

def laplacian(frame):
    img = cv2.Laplacian(frame, cv2.CV_64F)
    return img

def sobelX(frame, kernel_size):
    img = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=kernel_size)
    return img

def sobelY(frame, kernel_size):
    img = cv2.Sobel(frame, cv2.CV_64F, 0,1,ksize=kernel_size)
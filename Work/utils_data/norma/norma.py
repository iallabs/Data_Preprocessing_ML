import tensorflow as tf


def normalize(image, label):
  """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  return image, label
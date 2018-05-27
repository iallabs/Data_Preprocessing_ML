import tensorflow as tf
from image_pro.parameters import *
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

# The lower bound for the smallest side of the image for aspect-preserving
# resizing. For example, if an image is 500 x 1000, it will be resized to
# _RESIZE_MIN x (_RESIZE_MIN * 2).
_RESIZE_MIN = 256

def _decode_crop_and_flip_b(image_buffer,bbox, num_channels):
    """Crops the given image to a random part of the image, and randomly flips.
    We use the fused decode_and_crop op, which performs better than the two ops
    used separately in series, but note that this requires that the image be
    passed in as an un-decoded string Tensor.
    Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    num_channels: Integer depth of the image buffer for decoding.
    Returns:
    3-D tensor with cropped image.
    """
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Reassemble the bounding box in the format the crop op requires.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

  # Use the fused decode and crop op here, which is faster than each in series.
    cropped = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=num_channels)

  # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped





def _mean_image_subtraction(image, means, num_channels):
    """Subtracts the given means from each image channel.
    For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
        image: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.
        num_channels: number of color channels in the image that will be distorted.
    Returns:
        the centered image.
    Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)

    return image - means


def _smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
        height: an int32 scalar tensor indicating the current height.
        width: an int32 scalar tensor indicating the current width.
        resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    Returns:
        new_height: an int32 scalar tensor indicating the new height.
        new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

  # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

  # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.
    Args:
        image: A 3-D image `Tensor`.
        resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    Returns:
        resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.
    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.
    Args:
        image: A 3-D image `Tensor`.
        height: The target height for the resized image.
        width: The target width for the resized image.
    Returns:
        resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    return tf.image.resize_images(image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

def _contrast(image):
    image = tf.image.random_contrast(image, 0, 5)
    return image

def _brightness(image):
    image = tf.image.random_brightness(image, 1)
    return image

def _central_crop(image):
    return tf.image.central_crop(image, 0.9)


def preprocess_image_train(image_buffer,label, output_height=N_HEIGHT, output_width=N_WIDTH,
                     num_channels=T_Channnels):
    """Preprocesses the given image.
    Preprocessing includes decoding, cropping, and resizing for both training
    and eval images. Training preprocessing, however, introduces some random
    distortion of the image to improve accuracy.
    Args:
        image_buffer: scalar string Tensor representing the raw JPEG image buffer.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        num_channels: Integer depth of the image buffer for decoding.
        is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
    Returns:
        A preprocessed image.
    """
    # For training, we want to randomize some of the distortions.
    """image= _brightness(image_buffer)"""
    image = tf.image.central_crop(image_buffer, 0.875)
    """mage = _resize_image(image, output_height, output_width)"""
    

    return image,label


def preprocess_image_eval(image_buffer,label, output_height=N_HEIGHT, output_width=N_WIDTH,
                     num_channels=T_Channnels):
    """Preprocesses the given image.
    Preprocessing includes decoding, cropping, and resizing for both training
    and eval images. Training preprocessing, however, introduces some random
    distortion of the image to improve accuracy.
    Args:
        image_buffer: scalar string Tensor representing the raw JPEG image buffer.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        num_channels: Integer depth of the image buffer for decoding.
        is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
    Returns:
        A preprocessed image.
    """
    # For validation, we want to decode, resize, then just crop the middle.
    
    image = _aspect_preserving_resize(image_buffer, _RESIZE_MIN)
    image = tf.image.central_crop(image, 0.875)
    """image = _resize_image(image, output_height, output_width)"""
    

    return image, label



def preprocess_image_box(image_buffer, bbox, output_height, output_width,
                     num_channels, is_training=False):
    """Preprocesses the given image.
    Preprocessing includes decoding, cropping, and resizing for both training
    and eval images. Training preprocessing, however, introduces some random
    distortion of the image to improve accuracy.
    Args:
        image_buffer: scalar string Tensor representing the raw JPEG image buffer.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        num_channels: Integer depth of the image buffer for decoding.
        is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
    Returns:
        A preprocessed image.
    """
    if is_training:
    # For training, we want to randomize some of the distortions.
        image = _decode_crop_and_flip(image_buffer, bbox, num_channels)
        image = _resize_image(image, output_height, output_width)
    else:
    # For validation, we want to decode, resize, then just crop the middle.
        image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, output_height, output_width)

    image.set_shape([output_height, output_width, num_channels])

    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)
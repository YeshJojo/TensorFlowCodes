from __future__ import absolute_import, division, print_function
from IPython.display import Image as IImage, display
import numpy as np
import PIL
from PIL import Image
import random
import requests
import tensorflow as tf

# d = requests.get("https://www.paristoolkit.com/Images/xeffel_view.jpg.pagespeed.ic.8XcZNqpzSj.jpg")
# with open("image.jpeg", "wb") as f:
#    f.write(d.content)

# img = PIL.Image.open('image.jpeg')
img = PIL.Image.open('C:\python_env\TensorFlowCodes\Sign.jpg')
img.load()
img_array = np.array(img)
PIL.Image.fromarray(img_array)


def random_flip_left_right(image):
    return tf.image.random_flip_left_right(image)


PIL.Image.fromarray(random_flip_left_right(img_array).numpy()).save("random_flip_left_right.png")


def random_contrast(image, minval=0.6, maxval=1.4):
    r = tf.random.uniform([], minval=minval, maxval=maxval)
    image = tf.image.adjust_contrast(image, contrast_factor=r)
    return tf.cast(image, tf.uint8)


PIL.Image.fromarray(random_contrast(img_array).numpy()).save("random_contrast.png")


def random_brightness(image, minval=0., maxval=.2):
    r = tf.random.uniform([], minval=minval, maxval=maxval)
    image = tf.image.adjust_brightness(image, delta=r)
    return tf.cast(image, tf.uint8)


PIL.Image.fromarray(random_brightness(img_array).numpy())


def random_saturation(image, minval=0.4, maxval=2.):
    r = tf.random.uniform((), minval=minval, maxval=maxval)
    image = tf.image.adjust_saturation(image, saturation_factor=r)
    return tf.cast(image, tf.uint8)


PIL.Image.fromarray(random_saturation(img_array).numpy()).save("random_saturation.png")


def random_hue(image, minval=-0.04, maxval=0.08):
    r = tf.random.uniform((), minval=minval, maxval=maxval)
    image = tf.image.adjust_hue(image, delta=r)
    return tf.cast(image, tf.uint8)


PIL.Image.fromarray(random_hue(img_array).numpy()).save("random_hue.png")


def transform_image(image):
    image = random_flip_left_right(image)
    image = random_contrast(image)
    image = random_brightness(image)
    image = random_hue(image)
    image = random_saturation(image)
    return image


transformed_img = transform_image(img_array).numpy()
PIL.Image.fromarray(transformed_img)


def resize_image(image):
    image = tf.image.resize(image, size=(256, 256), preserve_aspect_ratio=False)
    image = tf.cast(image, tf.uint8)
    return image


PIL.Image.fromarray(resize_image(transformed_img).numpy())

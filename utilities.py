import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
from random import choice
import matplotlib.image as mpimg
import tensorflow as tf

global img_height
global img_width
global batch_size
batch_size = 32
img_height = 400
img_width = 400

def get_class_names(data_dir):
    global class_names  
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    return class_names  

def plot_ex_images(data_dir):
    plt.figure(figsize=(10, 10))
    for num, class_n in enumerate(class_names):
        files = Path.joinpath(data_dir, class_n).glob('*jpg')
        ran_image = choice(list(files))
        ax = plt.subplot(3, 4, num + 1)
        image = mpimg.imread(ran_image)
        plt.imshow(image)
        plt.title(class_n)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.grid(True)

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def configure_for_performance(ds, AUTOTUNE):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


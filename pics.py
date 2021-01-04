import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib


import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10

# fix random seed for reproducibility
seed = 21
numpy.random.seed(seed)

# load data
data_dir = pathlib.Path('/home/rey/ML/data/test')

batch_size = 32
img_height = 400
img_width = 400

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)
num_classes = len(class_names)

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)

val_ds = list_ds.take(val_size)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  print(one_hot)
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
  print(img)
  print(label)
  x = input('yo')
  return img, label



# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())


def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)


image_batch, label_batch = next(iter(train_ds))

print(label_batch)
plt.figure(figsize=(10, 10))
used = []
i = 0
j = 0
while i < 11:
  print(j)
  label = label_batch[j]
  print(class_names[label])
  if class_names[label] not in used:
      used.append(class_names[label])
      ax = plt.subplot(3, 4, i + 1)
      plt.imshow(image_batch[j].numpy().astype("uint8"), interpolation = 'none')
      print(label)
      plt.title(class_names[label])
      ax.axes.xaxis.set_visible(False)
      ax.axes.yaxis.set_visible(False)
      plt.grid(True)
      i += 1
  j += 1
plt.show()


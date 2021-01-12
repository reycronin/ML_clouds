import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
from random import choice
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D

global INPUT_SHAPE 
global batch_size
img_height = 400
img_width = 400
INPUT_SHAPE = (img_height, img_width, 3)
batch_size = 32

def get_class_names(data_dir):
	global class_names	
	class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
	num_classes = len(class_names)
	print('with', num_classes, 'different classes\n', class_names, '\n')
	return class_names

def load_data(data_dir):
	image_count = len(list(data_dir.glob('*/*.jpg')))
	print('there are', image_count, 'images\n')
	list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=True)
	list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=True)
	return list_ds, image_count 

def allocate_data(list_ds, image_count):
	# 20 percent of the data is allocated to the validation set
	val_size = int(image_count * 0.2)
	train_ds = list_ds.skip(val_size)
	val_ds = list_ds.take(val_size)
	print(tf.data.experimental.cardinality(train_ds).numpy(), 'images are used for training')
	print(tf.data.experimental.cardinality(val_ds).numpy(), 'images are used for validation')
	return train_ds, val_ds
	
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
	return tf.image.resize(img, [INPUT_SHAPE[0], INPUT_SHAPE[1])

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

def configuration(train_ds, val_ds):
	# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
	val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

	train_ds = configure_for_performance(train_ds, AUTOTUNE)
	val_ds = configure_for_performance(val_ds, AUTOTUNE)
	return train_ds, val_ds

def build_base_model():
	model = tf.keras.Sequential([
						layers.experimental.preprocessing.Rescaling(1./255)])
	model.add(Conv2D(filters=32, 
										kernel_size=3, 
										activation='relu',
										input_shape=INPUT_SHAPE))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Conv2D(filters=32, 
										kernel_size=3, 
										activation='relu',
										input_shape=INPUT_SHAPE))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Conv2D(filters=32, 
										kernel_size=3, 
										activation='relu',
										input_shape=INPUT_SHAPE))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Flatten()) #3D feature map to 1D feature vectors
	model.add(Dense(units = 128, activation='relu')),
	model.add(Dense(units = len(class_names)))

	model.compile(
		optimizer='adam',
		loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])

	return model






import time
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.models import load_model
import os
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from random import choice
import matplotlib.image as mpimg
import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.models import save_model, load_model


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
    plt.figure(figsize=(12, 12))
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

def label_histogram(data_dir):
    plt.figure(figsize=(10, 7.5))
    indices = len(list(class_names))
    file_count = []
    for class_n in class_names:
        files = Path.joinpath(data_dir, class_n).glob('*jpg')
        file_count.append(len(list(files)))
    plt.bar(list(class_names), file_count, color='seagreen')
    plt.ylabel('number of images', fontsize=14)
    plt.xlabel('label', fontsize=14)
    plt.tight_layout()
    plt.show()

def info_table():
    import plotly.graph_objects as go
    df = pd.read_csv('descriptions.csv', sep='; ', engine='python')
    df.style.set_properties(subset=['Symbol'], **{'width': '200px'})
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='lavender',
                    align='left'),
        cells=dict(values=[df.Symbol, df.Full_Name, df.Description],
                   fill_color='lavenderblush',
                   align='left'))
    ])
    fig.show()

def model_prediction(model, image_path):
    image = tf.keras.preprocessing.image.load_img(image_path)
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    model_predict_num = np.argmax(predictions[0])
    df = pd.read_csv('descriptions.csv', sep='; ', engine='python')
    model_predict = df.iloc[[model_predict_num]]['Symbol']
    model_predict = model_predict.to_string(index=False)
    print('The model predicts that these clouds are:', model_predict)
    return model_predict

def compare_yourself(data_dir, current_score, model):
    plt.figure(figsize=(12, 12))
    class_n = choice(list(class_names))
    print(class_n)
    files = Path.joinpath(data_dir, class_n).glob('*jpg')
    ran_image = choice(list(files))
    image = mpimg.imread(ran_image)
    plt.imshow(image)
    plt.title('which class does this image belong to?')
    plt.show()
    guess = input('define the class: ')
    current_score[1] += 1

    if guess.lower() == class_n.lower():
        print('CORRECT!')
        current_score[0] += 1
    else:
        print('not quite')
        print('correct answer: ', class_n)
    model_predict = model_prediction(model, ran_image)
    print('your current score is: ', current_score[0]/current_score[1])
    return(current_score)

def label_histogram(data_dir):
    plt.figure(figsize=(10, 7.5))
    indices = len(list(class_names))
    file_count = []
    for class_n in class_names:
        files = Path.joinpath(data_dir, class_n).glob('*jpg')
        file_count.append(len(list(files)))
    plt.xlabel('class name')
    plt.ylabel('image count')
    plt.bar(list(class_names), file_count, color='seagreen')

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
    return tf.image.resize(img, [INPUT_SHAPE[0], INPUT_SHAPE[1]])

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

def data_aug(train_ds):
    # the Sequential option groups linearly stacks the layers
    # rescale images to be 0 to 1

    IMG_SIZE = 400
    resize_and_rescale = tf.keras.Sequential([
      layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
      layers.experimental.preprocessing.Rescaling(1./255)])

    data_augmentation = tf.keras.Sequential([
      layers.experimental.preprocessing.RandomFlip("horizontal"),
      layers.experimental.preprocessing.RandomRotation(0.02),
      layers.experimental.preprocessing.RandomZoom(0.1)])
    model = tf.keras.Sequential([resize_and_rescale,
                                data_augmentation])

    plt.figure(figsize=(10,10))
    for images, _ in train_ds.take(1):
        ax = plt.subplot(3, 3, 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
        plt.title('unaugmented image')        
        for i in range(2, 10):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    return model

def build_base_model_dropout(model):

    # 2D convolution layer
    model.add(Conv2D(filters=32, # number of output filters in the convolution
                    kernel_size=3, # dimensions of the 2D convolution window
                    activation='relu', # activation function
                    input_shape=INPUT_SHAPE)) # shape of image input

    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=32,
                    kernel_size=3,
                    activation='relu',
                    input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=32,
                    kernel_size=3,
                    activation='relu',
                    input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))
    # 400 x 400 -> 16,000 pixels
    model.add(Flatten()) #2D feature map to 1D feature vectors
    # densely connected neural layer with 128 neurons
    model.add(Dense(units = 128, activation='relu')),
    model.add(Dropout(rate=0.2))
    # this returns nodes for scoring the classification of the current image
    # into each of the classes
    model.add(Dense(units = len(class_names), activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model




def fit_model(train_ds, val_ds, model, num_epochs):

    do_save = input('would you like to save your model? [y/n]: ')
    logger.info("Start training")
    search_start = time.time()
    history = model.fit(train_ds,
                         validation_data=val_ds,
                         epochs=num_epochs,
                         verbose=0
                     )
    search_end = time.time()
    elapsed_time = search_end - search_start
    logger.info(f"Elapsed time (s): {elapsed_time}")
    if do_save.lower() == 'y' or do_save.lower() == 'yes':
        save_model(model, 'cloud_model_' + str(num_epochs), overwrite=True, include_optimizer=True)
    return history, model

def load_existing_model():
    use = input('would you like to load a saved model? [y/n] ')
    if use.lower() == 'y' or use.lower() == 'yes':
        model_dir = input('input the name of the model directory: ')
        loaded_model = load_model(model_dir, custom_objects=None, compile=False)
        print('using the ' + model_dir + ' model')
        return(loaded_model)
    else:
        print('using existing model')

def plot_results(history, num_epochs):
    epochs_range = range(num_epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('percentage')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('epoch')
    plt.ylabel('percentage')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss - Baseline')

    plt.savefig('training_val_acc_loss' + str(num_epochs) + 'BASELINE.png')
    plt.show()

def show_bad_images(data_dir):
    bad_img = ['/Ac/Ac-N036.jpg', '/Ac/Ac-N144.jpg', '/Ac/Ac-N198.jpg' , '/As/As-N054.jpg', '/As/As-N148.jpg', '/Cc/Cc-N079.jpg', '/Cc/Cc-N109.jpg', '/Ac/Ac-N171.jpg']
    title = ['Ac taken from plane',
            'Ac that looks more like Cb',
            'Ac that doesnt really look like a cloud',
            'As that is only a cloud edge',
            'As that also has cumulus clouds in the picture',
            'Cc that also has Ct in the picture',
            'Cc picture that is also classified as Ac',
            'Ac picture that is also classified as Cc']

    plt.figure(figsize=(20, 12))
    for idx, img in enumerate(bad_img):
        ax = plt.subplot(2, 4, idx + 1) 
        img_path = Path(str(data_dir) + img)
        image = mpimg.imread(img_path)
        plt.imshow(image)
        plt.title(title[idx])
        plt.xticks([])
        plt.yticks([])
    plt.show()


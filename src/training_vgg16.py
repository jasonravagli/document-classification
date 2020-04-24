# -*- coding: utf-8 -*-
"""Document Classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R0O6BY8tXgJCzoY2JzamdCf8QCbXqxqa

# Setup

Install required packages:
"""

"""Import required packages:"""

import os
import h5py
import numpy as np
import tensorflow as tf
import keras
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
from tqdm import tqdm
import confuse
import logging
import configuration as config

"""General parameters and settings:"""

original_dataset_path = os.path.join(config.config["paths"]["root"].get(), config.config["paths"][
    "original_dataset"].get())  # "/content/drive/My Drive/document-classification/datasets/rvl-cdip/mini-dataset-1600-200-200.h5"
cnn_dataset_path = os.path.join(config.config["paths"]["root"].get(), config.config["paths"][
    "cnn_dataset"].get())  # "/content/drive/My Drive/document-classification/datasets/rvl-cdip/cnn-mini-dataset-1600-200-200.h5"
cnn_model_path = os.path.join(config.config["paths"]["root"].get(), config.config["paths"][
    "cnn_model"].get())  # "/content/drive/My Drive/document-classification/models/model.json"
cnn_weights_path = os.path.join(config.config["paths"]["root"].get(), config.config["paths"][
    "cnn_weights"].get())  # "/content/drive/My Drive/document-classification/models/weights.h5"
cnn_weights_checkpoint_path = os.path.join(config.config["paths"]["root"].get(), config.config["paths"][
    "cnn_weights_checkpoint"].get())  # "/content/drive/My Drive/document-classification/checkpoints/checkpoint_weights.h5"
history_stats_path = os.path.join(config.config["paths"]["root"].get(), config.config["paths"][
    "history_stats"].get())  # "/content/drive/My Drive/document-classification/history/history_stats.csv"

# Flag to generate CNN dataset (set it to False if the dataset has already been generated)
generate_dataset = config.config["generate_dataset"].get(bool)
# False: skip training process and load the already learned weights to only evaluate the network
train_network = config.config["train_network"].get(bool)
# True: resume the training from a checkpoint; False: start a new training
# This flag is checked only if train_network=True
resume_training = config.config["resume_training"].get(bool)

# Shape of the images read from the mini-dataset
original_img_shape = config.config["original_img_shape"].get()  # (1000, 750)
# Shape of the images in input to the CNN
cnn_image_shape = config.config["cnn_image_shape"].get()  # (500, 375)
# Number of images classes
n_classes = config.config["n_classes"].get(int)  # 16

learning_rate = config.config["training"]["lr"].get(float)
batch_size = config.config["training"]["batch_size"].get(int)

print(learning_rate)
print(batch_size)

"""Check if GPU is available:"""

# from tensorflow.python.client import device_lib
# logging.info(device_lib.list_local_devices())
assert tf.test.is_gpu_available()

"""# Dataset preprocessing

**Function to generate a dataset for the CNN (VGG16) starting from the corresponding original dataset.**

It reads all images from the original dataset, preprocesses them according to the CNN requirements (resize, convert to RGB) and finally saves them into the new dataset.
"""


def generate_dataset_for_cnn(hdf_original_file, hdf_cnn_file, dataset_name):
    # Number of data rows to read from original file, process and write to the CNN file at the same time
    # (To speedup read/write operations)
    batch_size = 64

    # Read datasets from the original HDF5 file
    ds_original_imgs = hdf_original_file[dataset_name]
    ds_original_labels = hdf_original_file[dataset_name + "_labels"]

    n_imgs = ds_original_imgs.len()

    # Create omonymous datasets into the HDF5 file for the CNN
    ds_cnn_imgs = hdf_cnn_file.create_dataset(dataset_name, (n_imgs, cnn_image_shape[0], cnn_image_shape[1], 3), dtype="float32", compression="gzip")
    ds_cnn_labels = hdf_cnn_file.create_dataset(dataset_name + "_labels", (n_imgs, n_classes), dtype="int8", compression="gzip")

    n_batches = n_imgs // batch_size
    for i in tqdm(range(n_batches)):
        # Load data in batches
        start_batch_index = i * batch_size
        end_batch_index = np.min([(i + 1) * batch_size, n_imgs])
        original_batch_imgs = ds_original_imgs[start_batch_index:end_batch_index]
        original_batch_labels = ds_original_labels[start_batch_index:end_batch_index]

        cnn_batch_imgs = np.empty((batch_size, cnn_image_shape[0], cnn_image_shape[1], 3), dtype=np.float32)
        # Labels do not need to be processed
        cnn_batch_labels = original_batch_labels

        # Preprocess all images in batch
        for i in range(batch_size):
            original_img = original_batch_imgs[i]
            # De-normalization: bring back pixel values to the [0,255] range
            non_normalized_img = np.asarray(original_img * 255, dtype=np.uint8)
            # Scale the image to the target shape (height/width ratio is preserved due
            # to the original and target shape values)
            scaled_img = cv2.resize(non_normalized_img, (cnn_image_shape[1], cnn_image_shape[0]), \
                                    interpolation=cv2.INTER_AREA)
            # Convert grayscale image to RGB (VGG16 needs color images)
            rgb_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGB)
            # Normalization
            cnn_batch_imgs[i] = rgb_img / 255

        # Write data batches
        ds_cnn_imgs[start_batch_index:end_batch_index] = cnn_batch_imgs
        ds_cnn_labels[start_batch_index:end_batch_index] = cnn_batch_labels


"""For each set (train. validation, test) in the original HDF5 dataset file, generate a set in a new HDF5 file to be used during the CNN training phase:"""

if generate_dataset:
    with h5py.File(original_dataset_path, 'r') as hdf_original_file, h5py.File(cnn_dataset_path, 'w') as hdf_cnn_file:
        logging.info("Generating training dataset for CNN...")
        generate_dataset_for_cnn(hdf_original_file, hdf_cnn_file, "train")
        logging.info("Generating validation dataset for CNN...")
        generate_dataset_for_cnn(hdf_original_file, hdf_cnn_file, "valid")
        logging.info("Generating test dataset for CNN...")
        generate_dataset_for_cnn(hdf_original_file, hdf_cnn_file, "test")

"""#Utility classes

##Data generator definition

Class to handle large datasets (see https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly for further details)
"""


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, ds_imgs, ds_labels, batch_size=32, dim=(32, 32, 32), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.ds_labels = ds_labels
        self.ds_imgs = ds_imgs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ds_imgs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.ds_imgs.len())
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for batch_index, ds_index in enumerate(indexes):
            # Store sample
            X[batch_index, :] = self.ds_imgs[ds_index]

            # Store class
            y[batch_index, :] = self.ds_labels[ds_index]

        return X, y


"""## History recording

Callback class that records into a csv file the loss values and other performance measures at each training epoch
"""


class LossHistory(keras.callbacks.Callback):
    def __init__(self, file_path, resume_training):
        super(LossHistory, self).__init__()
        self.file_path = file_path
        self.resume_training = resume_training

    def on_train_begin(self, logs={}):
        # if we are starting a new training, create the csv file and the header row
        if not self.resume_training:
            header = np.asarray([["Train-Loss", "Train-Accuracy", "Validation-Loss", "Validation-Accuracy"]])
            np.savetxt(self.file_path, header, fmt='%s', delimiter=",")

    def on_epoch_end(self, epoch, logs=None):
        with open(self.file_path, "a") as csv_file:
            statistics = np.asarray([[logs.get("loss"), logs.get("accuracy"), \
                                      logs.get("val_loss"), logs.get("val_accuracy")]])
            np.savetxt(csv_file, statistics)


"""##Custom metrics"""

# class MulticlassTruePositives(tf.keras.metrics.Metric):
#     def __init__(self, name='multiclass_true_positives', **kwargs):
#         super(MulticlassTruePositives, self).__init__(name=name, **kwargs)
#         self.hit_per_class = self.add_weight(name='hit_per_class', shape=(1, n_classes) initializer='zeros')
#         self.n_entries_per_class = self.add_weight(name='n_entries_per_class', shape=(1, n_classes) initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         class_index = tf.argmax(y_pred, axis=1)
#         hit = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')

#         if hit:
#             tf.convert_to_tensor


#         values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
#         values = tf.cast(values, 'float32')
#         if sample_weight is not None:
#             sample_weight = tf.cast(sample_weight, 'float32')
#             values = tf.multiply(values, sample_weight)
#         self.true_positives.assign_add(tf.reduce_sum(values))

#     def result(self):
#         return self.true_positives

#     def reset_states(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.true_positives.assign(0.)

"""# CNN model definition

**For the following part refer to:** https://www.pyimagesearch.com/2019/06/24/change-input-shape-dimensions-for-fine-tuning-with-keras/

Use the VGG16 as base model for fine-tuning:
"""

# load VGG16, ensuring the head FC layer sets are left off, while at
# the same time adjusting the size of the input image tensor to the
# network
baseModel = keras.applications.VGG16(weights="imagenet", include_top=False,
                                     input_tensor=Input(shape=(cnn_image_shape[0], cnn_image_shape[1], 3)),
                                     pooling="max")

"""Add the custom structure (to be trained) on top of the (trained) convolutional part:"""

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = Dense(4096, activation="relu")(headModel)
headModel = Dense(4096, activation="relu")(headModel)
headModel = Dense(n_classes, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
cnn_model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

"""Compile the model basing on the configuration flags:"""

opt = Adam(lr=learning_rate)

# If we not want to train the network, load the already trained weights
if not train_network:
    cnn_model.load_weights(cnn_weights_path)
# If the training process has to be resumed load the checkpointed weights
elif resume_training:
    cnn_model.load_weights(cnn_weights_checkpoint_path)

cnn_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

"""Show model structure info:"""

logging.info("Summary for the whole model...")
logging.info(cnn_model.input_shape)
logging.info(cnn_model.output_shape)
logging.info(cnn_model.summary())

"""# Training

Save the network model into a json file and its weights in a separate HDF5 file:
"""


def save_cnn_to_file(cnn):
    # serialize model to JSON
    model_json = cnn.to_json()
    with open(cnn_model_path, 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn.save_weights(cnn_weights_path)
    logging.info("Model saved")


"""Train the network (if the corresponding flag is set) and save it into the Drive.

Save a model checkpoint after each improvement in accuracy with respect to the validation set.
"""

# Data generators params
data_gen_params = {'dim': (cnn_image_shape[0], cnn_image_shape[1]),
                   'batch_size': batch_size,
                   'n_classes': n_classes,
                   'n_channels': 3,
                   'shuffle': True}

if train_network:
    with h5py.File(cnn_dataset_path, 'r') as hdf_cnn_file:
        # Generators
        training_generator = DataGenerator(hdf_cnn_file["train"], hdf_cnn_file["train_labels"], **data_gen_params)
        validation_generator = DataGenerator(hdf_cnn_file["valid"], hdf_cnn_file["valid_labels"], **data_gen_params)

        checkpoint = ModelCheckpoint(
            cnn_weights_checkpoint_path)  # , monitor="val_accuracy", mode="max", save_best_only=True)
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100, restore_best_weights=True)
        loss_history = LossHistory(file_path=history_stats_path, resume_training=resume_training)

        # Train model on dataset
        history = cnn_model.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          use_multiprocessing=True, epochs=1000,
                                          workers=0, callbacks=[es, checkpoint, loss_history], verbose=1)

        logging.info("\nTraining completed")

    save_cnn_to_file(cnn_model)

"""Evaluate the model on the test set:"""

with h5py.File(cnn_dataset_path, 'r') as hdf_cnn_file:
    # Evaluate the trained model on the test set
    test_generator = DataGenerator(hdf_cnn_file["test"], hdf_cnn_file["test_labels"], **data_gen_params)
    score = cnn_model.evaluate_generator(test_generator, verbose=1)

    logging.info("\nAccuracy on test set: " + str(score))

with h5py.File(cnn_dataset_path, 'r') as hdf_cnn_file:
    ds_imgs = hdf_cnn_file["test"]
    ds_labels = hdf_cnn_file["test_labels"]
    n_test_imgs = ds_imgs.len()

    predicted_labels = cnn_model.predict(x=ds_imgs, batch_size=data_gen_params["batch_size"])

    ds_labels_explicit = np.argmax(ds_labels, axis=-1)
    predicted_labels_explicit = np.argmax(predicted_labels, axis=-1)
    logging.info(ds_labels_explicit.shape)
    logging.info(predicted_labels_explicit.shape)
    comparison_matrix = np.empty((ds_labels.shape[0], 2), dtype=np.int32)
    comparison_matrix[:, 0] = predicted_labels_explicit
    comparison_matrix[:, 1] = ds_labels_explicit
    logging.info(comparison_matrix.shape)
    logging.info(comparison_matrix)

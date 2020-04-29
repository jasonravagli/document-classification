"""
    INCOMPLETE SCRIPT
"""

import cv2
import os
import h5py
import logging
import zipfile
import numpy as np
from tqdm import tqdm

import configuration as config
import utils


def unzip_dataset():
    logging.debug("Unzipping dataset...")
    with zipfile.ZipFile(utils.get_path_from_property("mini_dataset_zip"), 'r') as zip_file:
        zip_file.extractall(utils.get_path_from_property("mini_dataset_folder"))
    logging.debug("Dataset unzipped")


def generate_dataset_for_cnn():
    mini_dataset_folder = utils.get_path_from_property("mini_dataset_folder")

    # If the dataset is not found try to unzip the zipped version
    if os.path.isfile(mini_dataset_folder):
        unzip_dataset()

    with h5py.File(os.path.join(mini_dataset_folder, "labels.h5"), 'r') as hdf_labels, h5py.File(utils.get_path_from_property("cnn_dataset"), 'w') as hdf_cnn_dataset:
        __write_dataset_to_file_for_cnn("train", hdf_cnn_dataset, hdf_labels)
        __write_dataset_to_file_for_cnn("valid", hdf_cnn_dataset, hdf_labels)
        __write_dataset_to_file_for_cnn("test", hdf_cnn_dataset, hdf_labels)


def __write_dataset_to_file_for_cnn(dataset_name, hdf_dest, hdf_labels):
    # Target shape to resize the images
    cnn_image_shape = config.config["cnn_image_shape"].get()
    n_classes = config.config["n_classes"].get()
    # Path to the folder containing the original mini-dataset
    mini_dataset_folder = utils.get_path_from_property("mini_dataset_folder")
    # Number of data rows to write to the CNN file at the same time (To speedup write operations)
    batch_size = 64

    # Read dataset containing labels and imgs path
    ds_labels = hdf_labels[dataset_name]

    n_imgs = ds_labels.len()

    # Create datasets into the HDF5 file for the CNN
    ds_cnn_imgs = hdf_dest.create_dataset(dataset_name, (n_imgs, cnn_image_shape[0], cnn_image_shape[1], 3), dtype="float32", compression="gzip")
    ds_cnn_labels = hdf_dest.create_dataset(dataset_name + "_labels", (n_imgs, n_classes), dtype="uint8", compression="gzip")

    # Matrix to calculate the mean value of each pixel (used for standardization process)
    mean_img = np.array((n_imgs, cnn_image_shape[0], cnn_image_shape[1], 3), dtype=np.float)

    n_batches = n_imgs // batch_size
    for i in tqdm(range(n_batches)):
        # Load data in batches
        start_batch_index = i * batch_size
        end_batch_index = np.min([(i + 1) * batch_size, n_imgs])
        original_batch_imgs = ds_original_imgs[start_batch_index:end_batch_index]
        original_batch_labels = ds_original_labels[start_batch_index:end_batch_index]

        cnn_batch_imgs = np.empty((batch_size, cnn_image_shape[0], cnn_image_shape[1], 3), dtype=np.float32)
        cnn_batch_labels = np.empty((batch_size, n_classes), dtype="uint8")

        # Preprocess all images in batch
        for j in range(batch_size):
            # Read image from disk
            img_relative_path, img_label = ds_labels[i*batch_size + j]
            img_abs_path = os.path.join(mini_dataset_folder, img_relative_path)
            original_img = cv2.imread(img_abs_path, cv2.IMREAD_GRAYSCALE)
            # Scale the image to target shape ignoring the possible stretching effect
            scaled_img = cv2.resize(original_img, (cnn_image_shape[1], cnn_image_shape[0]), interpolation=cv2.INTER_AREA)
            # Convert grayscale image to RGB (VGG16 needs color images)
            rgb_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGB)
            # Normalized image
            normalized_img = rgb_img/float(255)

            mean_img += normalized_img

            original_img = original_batch_imgs[j]
            # De-normalization: bring back pixel values to the [0,255] range
            non_normalized_img = np.asarray(original_img * 255, dtype=np.uint8)
            # Scale the image to the target shape (height/width ratio is preserved due
            # to the original and target shape values)
            scaled_img = cv2.resize(non_normalized_img, (cnn_image_shape[1], cnn_image_shape[0]), \
                                    interpolation=cv2.INTER_AREA)
            # Convert grayscale image to RGB (VGG16 needs color images)
            rgb_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGB)
            # Normalization
            normalized = rgb_img / 255
            mean = np.mean(normalized)
            cnn_batch_imgs[j] = normalized - mean

        # Write data batches
        ds_cnn_imgs[start_batch_index:end_batch_index] = cnn_batch_imgs
        ds_cnn_labels[start_batch_index:end_batch_index] = cnn_batch_labels

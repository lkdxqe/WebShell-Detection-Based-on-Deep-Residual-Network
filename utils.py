import cv2
import glob
import random
import numpy as np

isNormalize = True


def load_images_from_dir(image_dir, label):
    images_list = []
    labels_list = []
    image_files = glob.glob(image_dir)
    for imageFile in image_files:
        img = cv2.imread(imageFile, 0)
        if isNormalize:
            img = img / 255.0
            std = np.std(img)
            if std > 0:
                img = (img - np.mean(img)) / std
        images_list.append(img)
        labels_list.append(label)
    return images_list, labels_list


def load_data(train_black_dir, train_white_dir, test_black_dir, test_white_dir):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    images, labels = load_images_from_dir(train_black_dir, 1)
    train_images.extend(images)
    train_labels.extend(labels)

    images, labels = load_images_from_dir(train_white_dir, 0)
    train_images.extend(images)
    train_labels.extend(labels)
    combined = list(zip(train_images, train_labels))
    random.shuffle(combined)
    train_images, train_labels = zip(*combined)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print(f"train_images.shape:{train_images.shape}")

    images, labels = load_images_from_dir(test_black_dir, 1)
    test_images.extend(images)
    test_labels.extend(labels)

    images, labels = load_images_from_dir(test_white_dir, 0)
    test_images.extend(images)
    test_labels.extend(labels)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    print(f"test_images.shape:{test_images.shape}")

    return (train_images, train_labels), (test_images, test_labels)

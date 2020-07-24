import glob
import os
import PIL
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import random
import keras_preprocessing.image
from keras.applications.inception_v3 import preprocess_input

def load_and_crop_img(path, grayscale=False, color_mode='rgb', target_size=None,
                      interpolation='nearest'):


    interpolation, crop = interpolation.split(":") if ":" in interpolation else (interpolation, "none")

    if crop == "none":
        return keras_preprocessing.image.utils.load_img(path,
                                                        grayscale=grayscale,
                                                        color_mode=color_mode,
                                                        target_size=target_size,
                                                        interpolation=interpolation)

    img = keras_preprocessing.image.utils.load_img(path,
                                                   grayscale=grayscale,
                                                   color_mode=color_mode,
                                                   target_size=None,
                                                   interpolation=interpolation)

    crop_fraction = 0.875
    target_width = target_size[1]
    target_height = target_size[0]

    if target_size is not None:
        if img.size != (target_width, target_height):

            if crop not in ["center", "random"]:
                raise ValueError('Invalid crop method {} specified.', crop)

            if interpolation not in keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(interpolation,
                                            ", ".join(
                                                keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS.keys())))

            resample = keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

            width, height = img.size

            target_size_before_crop = (target_width / crop_fraction, target_height / crop_fraction)
            ratio = max(target_size_before_crop[0] / width, target_size_before_crop[1] / height)
            target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
            img = img.resize(target_size_before_crop_keep_ratio, resample=resample)

            width, height = img.size

            if crop == "center":
                left_corner = int(round(width / 2)) - int(round(target_width / 2))
                top_corner = int(round(height / 2)) - int(round(target_height / 2))
                return img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
            elif crop == "random":
                left_shift = random.randint(0, int((width - target_width)))
                down_shift = random.randint(0, int((height - target_height)))
                return img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))

    return img


keras_preprocessing.image.iterator.load_img = load_and_crop_img

root = "input"
train_csv = 'input/train.csv'
valid_csv = "input/val.csv"
test_csv = 'input/test.csv'

# Image features and the batch size for the ImgDataGenerator
batch_size = 64


traindf = pd.read_csv(train_csv, dtype=str)
testdf = pd.read_csv(test_csv, dtype=str)
valdf = pd.read_csv(valid_csv, dtype=str)

# Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3,
    preprocessing_function=preprocess_input
)



train_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=None,
    x_col="image_path",
    y_col="label",
    batch_size=batch_size,
    shuffle=True,
    target_size=(32,32),
    interpolation="lanczos",
)



validation_generator = datagen.flow_from_dataframe(
    dataframe=valdf,
    directory=None,
    x_col="image_path",
    y_col="label",
    batch_size=batch_size,
    shuffle=True,
    target_size=(32,32),
    preprocessing_function=preprocess_input,
    interpolation="lanczos",
)

# Also rescaling here for the test data generator as well.
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=None,
    x_col="image_path",
    y_col="label",
    batch_size=batch_size,
    shuffle=False,
    target_size=(32,32),
    interpolation="lanczos",
)

my_iter_t = train_generator.next()
print(my_iter_t[0].shape)

my_iter_te = test_generator.next()
print(my_iter_te[0].shape)

my_iter_v = validation_generator.next()
print(my_iter_v[0].shape)


def buildCsv() -> None:
    train_img_paths = []
    val_img_paths = []
    test_img_paths = []

    train_cls_ids = []
    val_cls_ids = []
    test_cls_ids = []

    train_cls_labels = []
    val_cls_labels = []
    test_cls_labels = []

    class_dirs = glob.glob(os.path.join(root, "indoorCVPR_09/Images", "*").replace("\\", "/"))
    classes = [os.path.basename(d) for d in class_dirs if os.path.isdir(d)]
    classes.sort()
    class2id = {classes[i]: i for i in range(len(classes))}

    trainval_file_names = sorted(
        set(open(os.path.join(root, "TrainImages.txt").replace("\\", "/")).read().splitlines())
    )
    test_file_names = set(
        open(os.path.join(root, "TestImages.txt").replace("\\", "/")).read().splitlines()
    )

    # Split rate is 4:1 * 80 train - 80 val
    for i, name in enumerate(trainval_file_names):
        path = os.path.join(root, "indoorCVPR_09/Images", name).replace("\\", "/")
        label = os.path.dirname(name)
        cls_id = class2id[label]

        if i % 5 == 4:
            # for validation
            val_img_paths.append(path)
            val_cls_ids.append(cls_id)
            val_cls_labels.append(label)
        else:
            # for training
            train_img_paths.append(path)
            train_cls_ids.append(cls_id)
            train_cls_labels.append(label)

    # for test
    for name in test_file_names:
        path = os.path.join(root, "indoorCVPR_09/Images", name).replace("\\", "/")
        label = os.path.dirname(name)
        cls_id = class2id[label]

        test_img_paths.append(path)
        test_cls_ids.append(cls_id)
        test_cls_labels.append(label)

    train_df = pd.DataFrame(
        {"image_path": train_img_paths, "class_id": train_cls_ids, "label": train_cls_labels},
        columns=["image_path", "class_id", "label"],
    )

    val_df = pd.DataFrame(
        {"image_path": val_img_paths, "class_id": val_cls_ids, "label": val_cls_labels},
        columns=["image_path", "class_id", "label"],
    )

    test_df = pd.DataFrame(
        {"image_path": test_img_paths, "class_id": test_cls_ids, "label": test_cls_labels},
        columns=["image_path", "class_id", "label"],
    )

    if not os.path.exists(root):
        os.mkdir(root)

    train_df.to_csv(os.path.join(root, "train.csv").replace("\\", "/"), index=None)
    val_df.to_csv(os.path.join(root, "val.csv").replace("\\", "/"), index=None)
    test_df.to_csv(os.path.join(root, "test.csv").replace("\\", "/"), index=None)

    print("Done")

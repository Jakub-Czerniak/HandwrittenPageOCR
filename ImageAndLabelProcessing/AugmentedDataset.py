import sys
import os
import numpy as np
from PIL import Image, ImageOps
import cv2 as cv
import random
from ImageAndLabelProcessing.BaseDatasetsProcessing import drop_splits_target_folders, create_splits_target_folders, save_labels_to_txt


def convert_to_opencv(image_pil):
    open_cv_image = np.array(image_pil)
    open_cv_image = cv.cvtColor(open_cv_image, cv.COLOR_RGB2BGR)
    return open_cv_image


def convert_to_pil(image_cv):
    image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    return image_pil


def random_augmentation(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    random_int = np.random.randint(0, 6)
    if random_int == 0:
        print(' ')
    elif random_int == 1:
        print(' ')
    elif random_int == 2:
        print(' ')
    elif random_int == 3:
        print(' ')
    elif random_int == 4:
        print(' ')
    elif random_int == 5:
        print(' ')
    elif random_int == 6:
        print(' ')
    else:
        sys.exit('Random number not in range')

    return image


def augment_folder(folder_path):
    for image_path in os.listdir(folder_path):
        path = os.path.join(folder_path, image_path)
        print(path)
        random_augmentation(path)



random.seed(65)

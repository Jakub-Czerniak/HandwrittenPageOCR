import shutil
import sys
import os
import numpy as np
from PIL import Image, ImageOps
import cv2 as cv
import random
from ImageAndLabelProcessing.BaseDatasetsProcessing import drop_splits_target_folders, create_splits_target_folders, save_labels_to_txt
from ImageAndLabelProcessing.PreprocessedDataset import rotate, read_base_labels, rotate_polygons, make_labels_and_bounding_boxes, save_word_images, save_image, copy_base_word_labels


def convert_to_opencv(image_pil):
    open_cv_image = np.array(image_pil)
    open_cv_image = cv.cvtColor(open_cv_image, cv.COLOR_RGB2BGR)
    return open_cv_image


def convert_to_pil(image_cv):
    image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    return image_pil


def copy_base_page_label(image_name, target_pages_folder, split_folder):
    base_image_path = os.path.join('../Data/Pages/Base/images/', split_folder, image_name)
    target_image_path = os.path.join(target_pages_folder, 'images', split_folder, image_name)
    shutil.copyfile(base_image_path, target_image_path)


def salt_and_pepper_noise(image, noise_percent=0.005, s_to_p_ratio=0.5):
    h, w, ch = image.shape

    salt_count = int(noise_percent * s_to_p_ratio * image.size)
    for i in range(salt_count):
        x = random.randint(0, w)
        y = random.randint(0, h)
        image[y][x] = 255

    pepper_count = int(noise_percent * (1-s_to_p_ratio) * image.size)
    for i in range(pepper_count):
        x = random.randint(0, w)
        y = random.randint(0, h)
        image[y][x] = 0

    return image


def resize_polygons(polygons, ratio):
    polygons = polygons * ratio
    return polygons


def random_augmentation(image_path):
    image = cv.imread(image_path)
    random_int = np.random.randint(0, 6)
    img_name = os.path.basename(os.path.normpath(image_path))
    org_label_path = os.path.join('../Data/Pages/Base/labels/all_org', img_name.replace('jpg', 'txt'))
    target_words_folder = '../Data/Words/Augmented/'
    target_pages_folder = '../Data/Pages/Augmented/'
    split_folder = os.path.normpath(image_path).split(os.path.sep)[-2]

    if random_int == 0:
        rotation = random.choice([i for i in range(-10, 10) if i not in range(-1, 1)])
        image, shifts_x1, shifts_y1, shifts_x2 = rotate(image, rotation)
        polygons = read_base_labels(org_label_path)
        polygons = rotate_polygons(polygons, shifts_x1, shifts_y1, shifts_x2)
        yolo_labels, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_labels_to_txt(image_path, yolo_labels, target_pages_folder, split_folder)
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name)

    elif random_int == 1:
        noise_percent = random.choice(np.linspace(0.004, 0.02, 16))
        s_to_p_ratio = random.choice(np.linspace(0.25, 0.75, 25))
        image = salt_and_pepper_noise(image, noise_percent, s_to_p_ratio)
        copy_base_page_label(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name)

    elif random_int == 2:
        ksize = random.randint(5, 50)
        image = cv.GaussianBlur(image, (ksize, ksize), 0)
        copy_base_page_label(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name)

    elif random_int == 3:
        size = random.randint(540, 720)
        height, width = image.shape[:2]
        if height >= width:
            ratio = size / float(height)
            dim = (int(width * ratio), size)
        else:
            ratio = size / float(width)
            dim = (size, int(height * ratio))
        image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        polygons = read_base_labels(org_label_path)
        polygons = resize_polygons(polygons, ratio)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name)

    elif random_int == 4:
        height, width, channels = image.shape
        x_removals_count = width//100
        y_removals_count = height//100
        for i in range(x_removals_count):
            row = random.randint(0, width-1)
            image[row] = 255
        for i in range(y_removals_count):
            column = random.randint(0, width-1)
            image[:, column] = 255
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name)

    elif random_int == 5:
        print(' ')
    elif random_int == 6:
        print(' ')
    else:
        sys.exit('Random number for augmentation not in range')

    return image


def augment_folder(folder_path):
    for image_path in os.listdir(folder_path):
        path = os.path.join(folder_path, image_path)
        print(path)
        random_augmentation(path)


random.seed(65)
drop_splits_target_folders('../Data/Pages/Augmented/')
drop_splits_target_folders('../Data/Words/Augmented/')
create_splits_target_folders('../Data/Pages/Augmented/')
create_splits_target_folders('../Data/Words/Augmented/')
copy_base_word_labels('../Data/Words/Augmented/')

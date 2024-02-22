import shutil
import sys
import os
import numpy as np
from PIL import Image, ImageEnhance
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
    label_name = image_name.replace('.jpg', '.txt')
    base_label_path = os.path.join('../Data/Pages/Base/labels/', split_folder, label_name)
    target_label_path = os.path.join(target_pages_folder, 'labels', split_folder, label_name)
    shutil.copyfile(base_label_path, target_label_path)


def copy_base_page_image(image_name, target_pages_folder, split_folder):
    base_image_path = os.path.join('../Data/Pages/Base/images/', split_folder, image_name)
    target_image_path = os.path.join(target_pages_folder, 'images', split_folder, image_name)
    shutil.copyfile(base_image_path, target_image_path)


def salt_and_pepper_noise(image, noise_percent=0.005, s_to_p_ratio=0.5):
    h, w, ch = image.shape

    salt_count = int(noise_percent * s_to_p_ratio * image.size)
    for i in range(salt_count):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        image[y][x] = 255

    pepper_count = int(noise_percent * (1-s_to_p_ratio) * image.size)
    for i in range(pepper_count):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        image[y][x] = 0

    return image


def resize_polygons(polygons, ratio):
    polygons = (np.asarray(polygons) * ratio).astype(int)
    return polygons


def random_augmentation(image_path):
    image = cv.imread(image_path)
    random_aug = np.random.choice(['rotation', 'noise', 'blur', 'downsizing', 'removing rows, columns', 'saturation', 'brightness', 'contrast'])
    img_name = os.path.basename(os.path.normpath(image_path))
    img_name_aug = img_name.replace('.jpg', '_aug.jpg')
    org_label_path = os.path.join('../Data/Pages/Base/labels/all_org', img_name.replace('jpg', 'txt'))
    target_words_folder = '../Data/Words/Augmented/'
    target_pages_folder = '../Data/Pages/Augmented/'
    split_folder = os.path.normpath(image_path).split(os.path.sep)[-2]

    print('Augmenting by: ' + random_aug + ' image: ' + img_name)
    if random_aug == 'rotation':
        rotation = random.choice([i for i in range(-25, 25) if i not in range(-5, 5)])
        image, shifts_x1, shifts_y1, shifts_x2 = rotate(image, rotation)
        polygons = read_base_labels(org_label_path)
        polygons = rotate_polygons(polygons, shifts_x1, shifts_y1, shifts_x2)
        yolo_labels, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_labels_to_txt(image_path, yolo_labels, target_pages_folder, split_folder, copy_image_to_split=True)
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'noise':
        noise_percent = random.choice(np.linspace(0.01, 0.08, 14))
        s_to_p_ratio = random.choice(np.linspace(0.25, 0.75, 25))
        image = salt_and_pepper_noise(image, noise_percent, s_to_p_ratio)
        copy_base_page_label(img_name, target_pages_folder, split_folder)
        copy_base_page_image(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'blur':
        ksize = 0
        while ksize % 2 == 0:
            ksize = random.randint(5, 50)
        image = cv.GaussianBlur(image, (ksize, ksize), 0)
        copy_base_page_label(img_name, target_pages_folder, split_folder)
        copy_base_page_image(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'downsizing':
        size = random.randint(420, 720)
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
        yolo_labels, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_labels_to_txt(image_path, yolo_labels, target_pages_folder, split_folder, copy_image_to_split=True)
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'removing rows, columns':
        height, width, channels = image.shape
        x_removals_count = width//100
        y_removals_count = height//100
        for i in range(x_removals_count):
            row = random.randint(0, height-1)
            color = random.randint(0, 255)
            rows = random.randint(0, 4)
            image[row:row+rows] = color
        for i in range(y_removals_count):
            column = random.randint(0, width-1)
            color = random.randint(0, 255)
            cols = random.randint(0, 4)
            image[:, column:column+cols] = color
        copy_base_page_label(img_name, target_pages_folder, split_folder)
        copy_base_page_image(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'saturation':
        image_pil = convert_to_pil(image)
        filter = ImageEnhance.Color(image_pil)
        enhance = random.choice(np.concatenate((np.linspace(0.3, 0.7, 10), np.linspace(1.3, 3, 10))))
        image_pil = filter.enhance(enhance)
        image = convert_to_opencv(image_pil)
        copy_base_page_label(img_name, target_pages_folder, split_folder)
        copy_base_page_image(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'brightness':
        image_pil = convert_to_pil(image)
        filter = ImageEnhance.Brightness(image_pil)
        enhance = random.choice(np.concatenate((np.linspace(0.3, 0.7, 10), np.linspace(1.3, 3, 10))))
        image_pil = filter.enhance(enhance)
        image = convert_to_opencv(image_pil)
        copy_base_page_label(img_name, target_pages_folder, split_folder)
        copy_base_page_image(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'contrast':
        image_pil = convert_to_pil(image)
        filter = ImageEnhance.Contrast(image_pil)
        enhance = random.choice(np.concatenate((np.linspace(0.3, 0.7, 10), np.linspace(1.3, 3, 10))))
        image_pil = filter.enhance(enhance)
        image = convert_to_opencv(image_pil)
        copy_base_page_label(img_name, target_pages_folder, split_folder)
        copy_base_page_image(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    else:
        sys.exit(random_aug + ' is not a valid augmentation')

    return image


def augmenting_folder(folder_path):
    img_count = len([name for name in os.listdir(folder_path)])
    img_counter = 0
    for image_path in os.listdir(folder_path):
        img_counter += 1
        path = os.path.join(folder_path, image_path)
        print(str(img_counter) + '/' + str(img_count) + ' in ' + folder_path)
        random_augmentation(path)


if __name__ == '__main__':
    random.seed(65)
    drop_splits_target_folders('../Data/Pages/Augmented/')
    drop_splits_target_folders('../Data/Words/Augmented/')
    create_splits_target_folders('../Data/Pages/Augmented/')
    create_splits_target_folders('../Data/Words/Augmented/')
    copy_base_word_labels('../Data/Words/Augmented/')
    augmenting_folder('../Data/Pages/Base/images/val')
    augmenting_folder('../Data/Pages/Base/images/train')
    augmenting_folder('../Data/Pages/Base/images/test')

import shutil
import sys
import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2 as cv
import random
from ImageAndLabelProcessing.BaseDatasetsProcessing import drop_folder, create_splits_target_folders
from ImageAndLabelProcessing.PreprocessedDataset import rotate, shear, read_base_labels, rotate_polygons, make_labels_and_bounding_boxes, save_image, copy_base_word_labels


def convert_to_opencv(image_pil):
    open_cv_image = np.array(image_pil)
    open_cv_image = cv.cvtColor(open_cv_image, cv.COLOR_RGB2BGR)
    return open_cv_image


def convert_to_pil(image_cv):
    image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    return image_pil


def copy_base_page_label_aug(image_name, target_pages_folder, split_folder):
    label_name = image_name.replace('.jpg', '.txt')
    label_name_aug = image_name.replace('.jpg', '_aug.txt')
    base_label_path = os.path.join('../Data/Pages/Base/', split_folder, 'labels', label_name)
    target_label_path = os.path.join(target_pages_folder, split_folder, 'labels', label_name_aug)
    shutil.copyfile(base_label_path, target_label_path)


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


def save_labels_to_txt_aug(image_path, labels, base_folder, split_folder):
    if not os.path.getsize(image_path):
        sys.exit('Image: ' + image_path + ' does not exist.')
    target_label_path = os.path.join(base_folder, split_folder, 'labels')
    txt_name = os.path.basename(os.path.normpath(image_path)).replace('.jpg', '_aug.txt')
    file_name = os.path.join(target_label_path, txt_name)
    with open(file_name, 'w') as file:
        for label in labels:
            file.write(label + '\n')


def save_word_images_aug(image, image_path, bounding_boxes, base_folder, split_folder):
    for index in range(len(bounding_boxes)):
        left = bounding_boxes[index][0]
        upper = bounding_boxes[index][1]
        right = bounding_boxes[index][2]
        lower = bounding_boxes[index][3]
        img_crop = image[upper:lower, left:right]
        img_name = os.path.basename(os.path.normpath(image_path)).replace('.jpg', '_' + str(index) + '_aug.jpg')
        path_img = os.path.join(base_folder, split_folder, 'images', img_name)
        cv.imwrite(path_img, img_crop)


def copy_base_word_labels_aug(base_folder):
    split_folders = ['train', 'test', 'val']
    for fol_name in split_folders:
        source_path = os.path.join('../Data/Words/Base/', fol_name, 'labels/labels.txt')
        target_path = os.path.join(base_folder, fol_name, 'labels/labels_aug.txt')
        file = open(f"{source_path}", "r").readlines()
        with open(target_path, 'a') as new_file:
            for line in file:
                line = line.split(' ')
                line[0] = line[0].replace('.jpg', '_aug.jpg')
                new_file.write(line[0] + ' ' + line[1])



def random_augmentation(image_path):
    image = cv.imread(image_path)
    random_aug = np.random.choice(['rotation', 'shear', 'noise', 'blur', 'downsizing', 'removing rows, columns', 'saturation', 'brightness', 'contrast'])
    img_name = os.path.basename(os.path.normpath(image_path))
    img_name_aug = img_name.replace('.jpg', '_aug.jpg')
    org_label_path = os.path.join('../Data/Pages/Base/all_org/labels', img_name.replace('jpg', 'txt'))
    target_words_folder = '../Data/Words/Augmented/'
    target_pages_folder = '../Data/Pages/Augmented/'
    split_folder = os.path.normpath(image_path).split(os.path.sep)[-3]

    print('Augmenting by: ' + random_aug + ' image: ' + img_name)
    if random_aug == 'rotation':
        rotation = random.choice([i for i in range(-25, 25) if i not in range(-2, 2)])
        image, shifts_x1, shifts_y1, shifts_x2 = rotate(image, rotation)
        polygons = read_base_labels(org_label_path)
        polygons = rotate_polygons(polygons, shifts_x1, shifts_y1, shifts_x2)
        yolo_labels, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_labels_to_txt_aug(image_path, yolo_labels, target_pages_folder, split_folder)
        save_word_images_aug(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'shear':
        polygons = read_base_labels(org_label_path)
        is_x = bool(random.randint(0, 1))
        if is_x:
            x_shear = random.choice(np.linspace(-0.5, 0.5, 10))
            image, shifts_x, _ = shear(image, x_shear, 0)
            polygons = rotate_polygons(polygons, shifts_x, None, None)
        else:
            y_shear = random.choice(np.linspace(-0.5, 0.5, 10))
            image, _, shifts_y = shear(image, 0, y_shear)
            polygons = rotate_polygons(polygons, None, shifts_y, None)
        yolo_labels, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_labels_to_txt_aug(image_path, yolo_labels, target_pages_folder, split_folder)
        save_word_images_aug(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'noise':
        noise_percent = random.choice(np.linspace(0.01, 0.08, 14))
        s_to_p_ratio = random.choice(np.linspace(0.25, 0.75, 25))
        image = salt_and_pepper_noise(image, noise_percent, s_to_p_ratio)
        copy_base_page_label_aug(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images_aug(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'blur':
        ksize = 0
        while ksize % 2 == 0:
            ksize = random.randint(5, 25)
        image = cv.GaussianBlur(image, (ksize, ksize), 0)
        copy_base_page_label_aug(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images_aug(image, image_path, bounding_boxes, target_words_folder, split_folder)
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
        save_labels_to_txt_aug(image_path, yolo_labels, target_pages_folder, split_folder)
        save_word_images_aug(image, image_path, bounding_boxes, target_words_folder, split_folder)
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
        copy_base_page_label_aug(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images_aug(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'saturation':
        image_pil = convert_to_pil(image)
        filter = ImageEnhance.Color(image_pil)
        enhance = random.choice(np.concatenate((np.linspace(0.3, 0.7, 10), np.linspace(1.3, 3, 10))))
        image_pil = filter.enhance(enhance)
        image = convert_to_opencv(image_pil)
        copy_base_page_label_aug(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images_aug(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'brightness':
        image_pil = convert_to_pil(image)
        filter = ImageEnhance.Brightness(image_pil)
        enhance = random.choice(np.concatenate((np.linspace(0.3, 0.7, 10), np.linspace(1.3, 3, 10))))
        image_pil = filter.enhance(enhance)
        image = convert_to_opencv(image_pil)
        copy_base_page_label_aug(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images_aug(image, image_path, bounding_boxes, target_words_folder, split_folder)
        save_image(image, target_pages_folder, split_folder, img_name_aug)

    elif random_aug == 'contrast':
        image_pil = convert_to_pil(image)
        filter = ImageEnhance.Contrast(image_pil)
        enhance = random.choice(np.concatenate((np.linspace(0.3, 0.7, 10), np.linspace(1.3, 3, 10))))
        image_pil = filter.enhance(enhance)
        image = convert_to_opencv(image_pil)
        copy_base_page_label_aug(img_name, target_pages_folder, split_folder)
        polygons = read_base_labels(org_label_path)
        _, bounding_boxes = make_labels_and_bounding_boxes(polygons, image.shape[1], image.shape[0])
        save_word_images_aug(image, image_path, bounding_boxes, target_words_folder, split_folder)
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
    drop_folder('../Data/Pages/Augmented/')
    drop_folder('../Data/Words/Augmented/')
    create_splits_target_folders('../Data/Pages/Augmented/')
    create_splits_target_folders('../Data/Words/Augmented/')
    copy_base_word_labels_aug('../Data/Words/Augmented/')
    augmenting_folder('../Data/Pages/Base/val/images')
    augmenting_folder('../Data/Pages/Base/train/images')
    augmenting_folder('../Data/Pages/Base/test/images')

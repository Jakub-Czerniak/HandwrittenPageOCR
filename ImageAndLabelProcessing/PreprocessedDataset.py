import shutil
import sys
import numpy as np
import math
import cv2 as cv
import os
from skimage.morphology import skeletonize
from skimage.util import invert
from multiprocessing import Pool
from itertools import repeat
from ImageAndLabelProcessing.BaseDatasetsProcessing import drop_splits_target_folders, create_splits_target_folders, save_labels_to_txt


def average_grayscale_conversion(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j][0] = image[i][j][0]/3 + image[i][j][1]/3 + image[i][j][2]/3
    image = image[:, :, 0]
    return image


def weighted_grayscale_conversion(image):
    # Grayscale = 0.299R + 0.587G + 0.114B
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j][0] = np.uint8(0.288 * image[i][j][2] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2])
    image = image[:, :, 0]
    return image


def normalization_grayscale(image):
    img_max = image.max()
    img_min = image.min()
    if not img_max == 255 and img_min == 0:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = 255 * (image[i][j] - img_min) / (img_max - img_min)
    return image.astype(np.uint8)


def shear(image, x_shear, y_shear):
    max_y_shear = round(abs(image.shape[1] * y_shear))
    max_x_shear = round(abs(image.shape[0] * x_shear))
    new_height = image.shape[0] + max_y_shear
    new_width = image.shape[1] + max_x_shear
    if image.ndim == 2:
        img_sheared = np.zeros(shape=(new_height, new_width))
    elif image.ndim == 3:
        img_sheared = np.zeros(shape=(new_height, new_width, image.shape[2]))
    else:
        sys.exit('Image in shear has 1 or more than 3 dimensions.')
    shifts_x = np.zeros(shape=(image.shape[0]), dtype=int)
    shifts_y = np.zeros(shape=(image.shape[1]), dtype=int)
    for i in range(image.shape[0]):
        if x_shear < 0:
            shifts_x[i] = int(round(i * x_shear + max_x_shear))
        elif x_shear > 0:
            shifts_x[i] = int(round(i * x_shear))
        else:
            break
    for j in range(image.shape[1]):
        if y_shear < 0:
            shifts_y[j] = int(round(j * y_shear) + max_y_shear)
        elif y_shear > 0:
            shifts_y[j] = int(round(j * y_shear))
        else:
            break
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_sheared[i + shifts_y[j]][j + shifts_x[i]] = image[i][j]

    return img_sheared, shifts_x, shifts_y


def rotate(image, angle):
    angle = math.radians(angle)
    shear_x = -math.tan(angle/2)
    shear_y = math.sin(angle)
    image, shifts_x1, _ = shear(image, shear_x, 0)
    image, _, shifts_y1 = shear(image, 0, shear_y)
    image, shifts_x2, _ = shear(image, shear_x, 0)

    return image, shifts_x1, shifts_y1, shifts_x2


def binarization(image, threshold):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > threshold:
                image[i][j] = 255
            else:
                image[i][j] = 0
    return image


def calculate_row_variation(skew, image):
    temp_image, _, _, _ = rotate(image, skew)
    sum_in_row = np.zeros(temp_image.shape[0], dtype=int)
    for i in range(temp_image.shape[0]):
        sum_in_row[i] = temp_image[i].sum()
    variation = np.var(sum_in_row)
    return variation


def projection_profile_skew(image, max_skew):
    skew_range = np.linspace(-max_skew, max_skew, 21)
    with Pool(4) as pool:
        results = pool.starmap(calculate_row_variation, zip(skew_range, repeat(image)))
    best_result_id = max(range(len(results)), key=results.__getitem__)
    rotation = skew_range[best_result_id]
    return rotation


def thresholding_otsu(image):
    image = normalization_grayscale(image)
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return image


def adaptive_thresholding(image):
    image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 251, 20)
    return image


def denoising(image):
    image = cv.fastNlMeansDenoising(src=image, h=30, templateWindowSize=7, searchWindowSize=21)
    return image


def read_base_labels(path):
    polygons = []
    lines = open(f"{path}", "r").readlines()
    for line in lines:
        line = line.split(' ')
        polygons.append([[max(int(line[1])-1, 0), max(int(line[3])-1, 0)], [max(int(line[5])-1, 0), max(int(line[7])-1, 0)], [max(int(line[9])-1, 0), max(int(line[11])-1, 0)], [max(int(line[13])-1, 0), max(int(line[15])-1, 0)]])
        # lu, ru, ld, rd
    return polygons


def rotate_polygons(polygons, rot_shifts_x1, rot_shifts_y1, rot_shifts_x2):
    for polygon in polygons:
        for i in range(4):
            polygon[i][0] += rot_shifts_x1[polygon[i][1]]
        for j in range(4):
            polygon[j][1] += rot_shifts_y1[polygon[j][0]]
        for k in range(4):
            polygon[k][0] += rot_shifts_x2[polygon[k][1]]
    return polygons


def make_labels_and_bounding_boxes(polygons, img_width, img_height):
    yolo_labels = []
    bounding_boxes = []
    for polygon in polygons:
        bounding_upper = min(polygon[0][1], polygon[1][1])
        bounding_down = max(polygon[2][1], polygon[3][1])
        bounding_left = min(polygon[0][0], polygon[2][0])
        bounding_right = max(polygon[1][0], polygon[3][0])
        if bounding_left > bounding_right:
            temp = bounding_right
            bounding_right = bounding_left
            bounding_left = temp
        if bounding_upper > bounding_down:
            temp = bounding_upper
            bounding_upper = bounding_down
            bounding_down = temp
        bb_width = (bounding_right - bounding_left) / img_width
        bb_height = (bounding_down - bounding_upper) / img_height
        x_center = (bounding_right - ((bounding_right - bounding_left) / 2)) / img_width
        y_center = (bounding_down - ((bounding_down - bounding_upper) / 2)) / img_height
        yolo_labels.append('word ' + str(x_center) + ' ' + str(y_center) + ' ' + str(bb_width) + ' ' + str(bb_height))
        bounding_boxes.append([bounding_left, bounding_upper, bounding_right, bounding_down])

    return yolo_labels, bounding_boxes


def save_word_images(image, image_path, bounding_boxes, base_folder, split_folder):
    for index in range(len(bounding_boxes)):
        left = bounding_boxes[index][0]
        upper = bounding_boxes[index][1]
        right = bounding_boxes[index][2]
        lower = bounding_boxes[index][3]
        img_crop = image[upper:lower, left:right]
        img_name = os.path.basename(os.path.normpath(image_path)).replace('.jpg', '_' + str(index) + '.jpg')
        path_img = os.path.join(base_folder, 'images', split_folder, img_name)
        cv.imwrite(path_img, img_crop)


def save_image(image, base_folder, split_folder, image_name):
    img_path = os.path.join(base_folder, 'images', split_folder, image_name)
    cv.imwrite(img_path, image)


def copy_base_word_labels(base_folder):
    split_folders = ['train', 'test', 'val']
    for fol_name in split_folders:
        source_path = os.path.join('../Data/Words/Base/labels', fol_name)
        target_path = os.path.join(base_folder, 'labels', fol_name)
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)


def preprocessing(image_path):
    img = cv.imread(image_path)
    img = weighted_grayscale_conversion(img)
    img = adaptive_thresholding(img)
    img = invert(img)
    img_skel = skeletonize(img).astype(np.uint8)
    img_skel = normalization_grayscale(img_skel)
    rotation = projection_profile_skew(img_skel, 5)
    img = skeletonize(img).astype(np.uint8)
    img, shifts_x1, shifts_y1, shifts_x2 = rotate(img, rotation)
    img = invert(img)
    img = normalization_grayscale(img)

    img_name = os.path.basename(os.path.normpath(image_path))
    org_label_path = os.path.join('../Data/Pages/Base/labels/all_org', img_name.replace('jpg', 'txt'))
    target_words_folder = '../Data/Words/Preprocessed/'
    target_pages_folder = '../Data/Pages/Preprocessed/'
    split_folder = os.path.normpath(image_path).split(os.path.sep)[-2]
    polygons = read_base_labels(org_label_path)
    polygons = rotate_polygons(polygons, shifts_x1, shifts_y1, shifts_x2)
    yolo_labels, bounding_boxes = make_labels_and_bounding_boxes(polygons, img.shape[1], img.shape[0])
    save_labels_to_txt(image_path, yolo_labels, target_pages_folder, split_folder, False)
    save_word_images(img, image_path, bounding_boxes, target_words_folder, split_folder)
    save_image(img, target_pages_folder, split_folder, img_name)


def preprocessing_folder(folder_path):
    img_count = len([name for name in os.listdir(folder_path)])
    img_counter = 0
    for image_path in os.listdir(folder_path):
        img_counter += 1
        path = os.path.join(folder_path, image_path)
        print(str(img_counter) + '/' + str(img_count) + ' in ' + folder_path)
        print('Preprocessing image: ' + path)
        preprocessing(path)


def preprocessing_sample(image_path):
    image_name = os.path.join('../Data/PreprocessedSample/', os.path.basename(os.path.normpath(image_path)))
    if not os.path.exists(image_name):
        os.makedirs(image_name)
    img = cv.imread(image_path)
    cv.imwrite(image_name + '/orginal.jpg', img)
    img = weighted_grayscale_conversion(img)
    cv.imwrite(image_name + '/grayscale.jpg', img)
    img = adaptive_thresholding(img)
    cv.imwrite(image_name + '/thresholding.jpg', img)
    img = invert(img)
    img_skel = skeletonize(img).astype(np.uint8)
    img_skel = normalization_grayscale(img_skel)
    rotation = projection_profile_skew(img_skel, 5)
    img = skeletonize(img).astype(np.uint8)
    img = normalization_grayscale(img)
    img = invert(img)
    cv.imwrite(image_name + '/skeletonize.jpg', img)
    img, _, _, _ = rotate(img, rotation)
    cv.imwrite(image_name + '/skew.jpg', img)


if __name__ == '__main__':
    drop_splits_target_folders('../Data/Pages/Preprocessed/')
    drop_splits_target_folders('../Data/Words/Preprocessed/')
    create_splits_target_folders('../Data/Pages/Preprocessed/')
    create_splits_target_folders('../Data/Words/Preprocessed/')
    copy_base_word_labels('../Data/Words/Preprocessed/')
    preprocessing_folder('../Data/Pages/Base/images/val')
    preprocessing_folder('../Data/Pages/Base/images/train')
    preprocessing_folder('../Data/Pages/Base/images/test')
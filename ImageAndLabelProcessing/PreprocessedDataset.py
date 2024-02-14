import shutil

import numpy as np
import math
import cv2 as cv
import os
from skimage.morphology import skeletonize
from skimage.util import invert
import sys
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
    return image


def shear(image, x_shear, y_shear):
    max_y_shear = round(abs(image.shape[1] * y_shear))
    max_x_shear = round(abs(image.shape[0] * x_shear))
    new_height = image.shape[0] + max_y_shear
    new_width = image.shape[1] + max_x_shear
    img_sheared = np.zeros(shape=(new_height, new_width))
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


def remove_black_borders(image):
    index_row_start = 0
    index_col_start = 0
    index_row_end = image.shape[0] - 1
    index_col_end = image.shape[1] - 1
    np.set_printoptions(threshold=sys.maxsize)
    while np.all(image[index_row_start] == 0):
        index_row_start += 1
    while np.all(image[index_row_end] == 0):
        index_row_end -= 1
    while np.all(image[:, index_col_start] == 0):
        index_col_start += 1
    while np.all(image[:, index_col_end] == 0):
        index_col_end -= 1
    image = image[index_row_start:index_row_end, index_col_start:index_col_end]
    return image


def binarization(image, threshold):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > threshold:
                image[i][j] = 255
            else:
                image[i][j] = 0
    return image


def projection_profile_skew(image, max_skew):
    max_variation = 0
    rotation = 0
    skew_range = np.linspace(-max_skew, max_skew, 21)
    for skew in skew_range:
        temp_image, _, _, _ = rotate(image, skew)
        sum_in_row = np.zeros(temp_image.shape[0], dtype=int)
        for i in range(temp_image.shape[0]):
            sum_in_row[i] = temp_image[i].sum()
        variation = np.var(sum_in_row)
        if max_variation < variation:
            max_variation = variation
            rotation = skew
        else:
            break
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


def preprocessing_folder(folder_path):
    for image in os.listdir(folder_path):
        preprocessing(folder_path + image)


def show_image(img):
    h = img.shape[0]
    w = img.shape[1]
    ratio = h / 480
    img = cv.resize(img, (480, int(w / ratio)))
    cv.imshow('img', img)
    cv.waitKey()


def read_base_labels(path):
    polygons = []
    lines = open(f"{path}", "r").readlines()
    for line in lines:
        line = line.split(' ')
        polygons.append([[int(line[1]), int(line[3])], [int(line[5]), int(line[7])], [int(line[9]), int(line[11])],
                       [int(line[13]), int(line[15])]])
        # lu, ru, ld, rd
    print(polygons)
    return polygons


def rotate_polygons(polygons, rot_shifts_x1, rot_shifts_y1, rot_shifts_x2):
    for polygon in polygons:
        for i in range(4):
            polygon[i][0] += rot_shifts_x1[polygon[i][0]]
        for j in range(4):
            polygon[j][1] += rot_shifts_y1[polygon[j][1]]
        for k in range(4):
            polygon[k][0] += rot_shifts_x2[polygon[k][0]]
    return polygons


def make_labels_and_bounding_boxes(polygons, img_width, img_height):
    yolo_labels = []
    bounding_boxes = []
    for polygon in polygons:
        bounding_upper = min(polygon[0][1], polygon[1][1])
        bounding_down = max(polygon[2][1], polygon[3][1])
        bounding_left = min(polygon[0][0], polygon[2][0])
        bounding_right = max(polygon[1][0], polygon[3][0])
        bb_width = (bounding_right - bounding_left) / img_width
        bb_height = (bounding_down - bounding_upper) / img_height
        x_center = (bounding_right - ((bounding_right - bounding_left) / 2)) / img_width
        y_center = (bounding_down - ((bounding_down - bounding_upper) / 2)) / img_height
        yolo_labels.append('word ' + str(x_center) + ' ' + str(y_center) + ' ' + str(bb_width) + ' ' + str(bb_height))
        bounding_boxes.append([bounding_left, bounding_upper, bounding_right, bounding_down])

    return yolo_labels, bounding_boxes


def save_word_image(image, image_path, bounding_boxes, base_folder, split_folder):
    for index in range(len(bounding_boxes)):
        left = bounding_boxes[index][0]
        upper = bounding_boxes[index][1]
        right = bounding_boxes[index][2]
        lower = bounding_boxes[index][3]
        img_crop = image[upper:lower, left:right]
        img_name = os.path.basename(os.path.normpath(image_path)).replace('.jpg', '_' + str(index) + '.jpg')
        path_img = os.path.join(base_folder, 'images', split_folder, img_name)
        cv.imwrite(path_img, img_crop)


def save_image(image, base_folder, split_folder):



def copy_base_word_labels(base_folder, split_folder):
    source_path = os.path.join('../Data/Words/Base/labels', split_folder)
    target_path = os.path.join(base_folder, '/labels', split_folder)
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
    img = normalization_grayscale(img)
    img, shifts_x1, shifts_y1, shifts_x2 = rotate(img, rotation)
    img = invert(img)

    org_label_path = ' '
    target_label_folder = ' '
    target_word_image_folder = ' '
    target_image_folder = ' '
    split_folder = ' '
    polygons = read_base_labels(org_label_path)
    polygons = rotate_polygons(polygons, shifts_x1, shifts_y1, shifts_x2)
    yolo_labels, bounding_boxes = make_labels_and_bounding_boxes(polygons, img.shape[0], img.shape[1])
    save_labels_to_txt(image_path, yolo_labels, target_label_folder, split_folder, False)
    save_word_image(img, image_path, bounding_boxes, target_word_image_folder, split_folder)
    save_image(img, target_image_folder, split_folder)


    """if not os.path.exists('PreprocessedImages/'):
        os.makedirs('PreprocessedImages/')
    image_name = os.path.basename(os.path.normpath(image_path))
    cv.imwrite('PreprocessedImages/' + image_name, img)"""


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



drop_splits_target_folders()
create_splits_target_folders()
copy_base_word_labels()
preprocessing_sample('../Data/Pages/Base/images/train/eng_AF_005.jpg')


import numpy as np
import math
import cv2 as cv
import os
from skimage.morphology import skeletonize
from skimage.util import invert
import sys


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
    temp = np.zeros(shape=(new_height, new_width))
    for i in range(image.shape[0]):
        if x_shear < 0:
            shift_x = round(i * x_shear + max_x_shear)
        elif x_shear > 0:
            shift_x = round(i * x_shear)
        else:
            shift_x = 0
        for j in range(image.shape[1]):
            if y_shear < 0:
                shift_y = round(j * y_shear) + max_y_shear
            elif y_shear > 0:
                shift_y = round(j * y_shear)
            else:
                shift_y = 0
            temp[i + shift_y][j + shift_x] = image[i][j]
    return temp


def rotate(image, angle):
    angle = math.radians(angle)
    shear_x = -math.tan(angle/2)
    shear_y = math.sin(angle)
    image = shear(image, shear_x, 0)
    image = shear(image, 0, shear_y)
    image = shear(image, shear_x, 0)
    image = remove_black_borders(image)
    return image


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
        temp_image = rotate(image, skew)
        sum_in_row = np.zeros(temp_image.shape[0], dtype=int)
        for i in range(temp_image.shape[0]):
            sum_in_row[i] = temp_image[i].sum()
        variation = np.var(sum_in_row)
        print('Variation: ' + str(variation) + ' Skew: ' + str(skew))
        if max_variation < variation:
            max_variation = variation
            rotation = skew
        else:
            break
    image = np.ascontiguousarray(rotate(image, rotation))
    print('Rotated by ' + str(rotation))
    return image


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
    # h = img.shape[0]
    # w = img.shape[1]
    # ratio = h / 480
    # img = cv.resize(img, (480, int(w / ratio)))
    cv.imshow('img', img)
    cv.waitKey()


def preprocessing(image_path):
    img = cv.imread(image_path)
    img = weighted_grayscale_conversion(img)
    img = adaptive_thresholding(img)
    img = invert(img)
    img = projection_profile_skew(img, 5)
    img = skeletonize(img).astype(np.uint8)
    img = normalization_grayscale(img)
    img = denoising(img)
    if not os.path.exists('PreprocessedImages/'):
        os.makedirs('PreprocessedImages/')
    image_name = os.path.basename(os.path.normpath(image_path))
    cv.imwrite('PreprocessedImages/' + image_name, img)


def preprocessing_sample(image_path):
    image_name = os.path.basename(os.path.normpath(image_path))
    if not os.path.exists(image_name):
        os.makedirs(image_name)
    img = cv.imread(image_path)
    cv.imwrite(image_name + '/orginal.jpg', img)
    img = denoising(img)
    cv.imwrite(image_name + '/noise.jpg', img)
    img = weighted_grayscale_conversion(img)
    cv.imwrite(image_name + '/grayscale.jpg', img)
    img = adaptive_thresholding(img)
    cv.imwrite(image_name + '/thresholding.jpg', img)
    img = invert(img)
    img = skeletonize(img).astype(np.uint8)
    img = normalization_grayscale(img)
    cv.imwrite(image_name + '/skeletonize.jpg', img)
    img = projection_profile_skew(img, 5)
    cv.imwrite(image_name + '/skew.jpg', img)


preprocessing_sample('SampleFiles/eng_AF_015.jpg')
preprocessing_sample('SampleFiles/eng_AF_019.jpg')
preprocessing_sample('SampleFiles/eng_AF_065.jpg')
preprocessing_sample('SampleFiles/eng_AS_051.jpg')
preprocessing_sample('SampleFiles/eng_AS_084.jpg')
preprocessing_sample('SampleFiles/eng_EU_126.jpg')
preprocessing_sample('SampleFiles/eng_EU_132.jpg')
preprocessing_sample('SampleFiles/eng_EU_143.jpg')

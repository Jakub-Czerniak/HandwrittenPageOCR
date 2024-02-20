import os
import shutil
import sys
import numpy as np
from PIL import Image, ImageOps

# unpack files from https://goodnotes.com/gnhk/ in sage format to main folder then run this script
# to split dataset, clean labels and create base word images from bounding boxes.


def move_tree(source_dir, target_dir):
    file_names = os.listdir(source_dir)
    for file in file_names:
        shutil.move(os.path.join(source_dir, file), target_dir)


def get_path_and_labels(sample_line, source_image_path): # process single image
    labels_yolo = []
    words_with_bb = []
    labels_org_sort = []
    points = np.zeros((4, 2))
    line_split = sample_line.translate({34: None, 44: None, 91: None, 93: None, 123: None, 125: None})
    line_split = line_split.split("text:")
    path = os.path.join(source_image_path, line_split[0].split(" ")[1])
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    width, height = img.size
    for i, line in enumerate(line_split[1:]):
        line = line.split(" ")
        points[0][0] = float(line[4])  # point x value
        points[0][1] = float(line[6])  # point y value
        points[1][0] = float(line[8])
        points[1][1] = float(line[10])
        points[2][0] = float(line[12])
        points[2][1] = float(line[14])
        points[3][0] = float(line[16])
        points[3][1] = float(line[18])
        sort_left = points[:, 0].argsort()
        points = points[sort_left, :]
        sort_upper_left = points[:2, 1].argsort()
        sort_upper_right = points[2:, 1].argsort() + 2
        sort_all = np.concatenate((sort_upper_left, sort_upper_right))
        points = points[sort_all, :]

        left_upper_x = int(points[0][0])
        left_upper_y = int(points[0][1])
        left_down_x = int(points[1][0])
        left_down_y = int(points[1][1])
        right_upper_x = int(points[2][0])
        right_upper_y = int(points[2][1])
        right_down_x = int(points[3][0])
        right_down_y = int(points[3][1])
        bounding_upper = min(left_upper_y, right_upper_y)
        bounding_down = max(left_down_y, right_down_y)
        bounding_left = min(left_down_x, left_upper_x)
        bounding_right = max(right_upper_x, right_down_x)
        bb_width = (bounding_right - bounding_left) / width
        bb_height = (bounding_down - bounding_upper) / height
        x_center = (bounding_right - ((bounding_right - bounding_left) / 2)) / width
        y_center = (bounding_down - ((bounding_down - bounding_upper) / 2)) / height
        if not (line[1] == '' or line[1][0] == '%'):
            labels_yolo.append('word ' + str(x_center) + ' ' + str(y_center) + ' ' + str(bb_width) + ' ' + str(bb_height))
            words_with_bb.append([line[1], bounding_left, bounding_upper, bounding_right, bounding_down])
            labels_org_sort.append('lu_x ' + str(left_upper_x) + ' lu_y ' + str(left_upper_y) + ' ru_x ' + str(right_upper_x) + ' ru_y ' + str(right_upper_y) + ' ld_x ' + str(left_down_x) + ' ld_y ' + str(left_down_y) + ' rd_x ' + str(right_down_x) + ' rd_y ' + str(right_down_y))

    return path, labels_yolo, labels_org_sort, words_with_bb


def drop_splits_target_folders(base_folder):
    path_labels = os.path.join(base_folder, 'labels')
    path_images = os.path.join(base_folder, 'images')
    if os.path.exists(path_labels) and os.path.isdir(path_labels):
        shutil.rmtree(path_labels)
    if os.path.exists(path_images) and os.path.isdir(path_images):
        shutil.rmtree(path_images)


def create_splits_target_folders(base_folder):
    path_labels = os.path.join(base_folder, 'labels')
    path_images = os.path.join(base_folder, 'images')
    if os.path.exists(path_labels) or os.path.exists(path_images):
        sys.exit('Target folder already exists.')
    folders = ['train', 'test', 'val']
    for f_name in folders:
        target_label_path = os.path.join(path_labels, f_name)
        target_image_path = os.path.join(path_images, f_name)
        os.makedirs(target_label_path)
        os.makedirs(target_image_path)


def save_labels_to_txt(image_path, labels, base_folder, split_folder, copy_image_to_split=False):
    if not os.path.getsize(image_path):
        sys.exit('Image: ' + image_path + ' does not exist.')
    target_label_path = os.path.join(base_folder, 'labels', split_folder)
    if copy_image_to_split:
        target_image_path = os.path.join(base_folder, 'images', split_folder)
        shutil.copy(image_path, target_image_path)
    txt_name = os.path.basename(os.path.normpath(image_path)).replace('jpg', 'txt')
    file_name = os.path.join(target_label_path, txt_name)
    with open(file_name, 'w') as file:
        for label in labels:
            file.write(label + '\n')


def save_word_image_label(image_path, labels, base_folder, split_folder):
    if not os.path.getsize(image_path):
        sys.exit('Image: ' + image_path + ' does not exist.')
    img_names = []
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    for index in range(len(labels)):
        left = labels[index][1]
        upper = labels[index][2]
        right = labels[index][3]
        lower = labels[index][4]
        img_crop = image.crop((left, upper, right, lower))
        img_name = os.path.basename(os.path.normpath(image_path)).replace('.jpg', '_' + str(index) + '.jpg')
        img_names.append(img_name)
        path_img = os.path.join(base_folder, 'images', split_folder, img_name)
        if img_crop.mode in ("RGBA", "P"):
            img_crop = img_crop.convert("RGB")
        img_crop.save(path_img)

    file_name = os.path.join(base_folder, 'labels', split_folder, 'labels.txt')
    with open(file_name, 'a') as file:
        assert len(img_names) == len(labels)
        for index in range(len(labels)):
            file.write(img_names[index] + ' ' + labels[index][0] + '\n')


if __name__ == '__main__':
    np.random.seed(65)

    source_image_paths = ["../train/", "../test/"]
    target_image_path = "../Data/Pages/Base"
    target_word_image_path = "../Data/Words/Base"

    if not (os.path.exists(source_image_paths[0]) and os.path.exists(source_image_paths[1]) and os.path.isdir(source_image_paths[0]) and os.path.isdir(source_image_paths[1])):
        sys.exit("Main folder does not contain data folders: " + source_image_paths[0] + ' ' + source_image_paths[1])

    os.makedirs(target_image_path, exist_ok=True)
    os.makedirs(target_word_image_path, exist_ok=True)

    move_tree(source_image_paths[1], source_image_paths[0])
    source_image_path = source_image_paths[0]
    shutil.rmtree(source_image_paths[1])

    source_manifest_paths = ["../train/train.manifest", "../train/test.manifest"]
    train_samples = open(f"{source_manifest_paths[0]}", "r").readlines()
    file_list = open(f"{source_manifest_paths[1]}", "r").readlines()
    np.random.shuffle(file_list)
    split_idx = int(0.5 * len(file_list))
    validation_samples = file_list[0:split_idx]
    test_samples = file_list[split_idx:]

    assert len(file_list) == len(validation_samples) + len(test_samples)

    print(f"Total training samples: {len(train_samples)}")
    print(f"Total validation samples: {len(validation_samples)}")
    print(f"Total test samples: {len(test_samples)}")

    drop_splits_target_folders(target_image_path)
    drop_splits_target_folders(target_word_image_path)
    create_splits_target_folders(target_image_path)
    create_splits_target_folders(target_word_image_path)
    os.makedirs(os.path.join(target_image_path, 'labels/all_org'))

    for i, line in enumerate(train_samples):
        image_path, yolo_labels, labels_org_sort, words = get_path_and_labels(line, source_image_path)
        save_labels_to_txt(image_path, yolo_labels, target_image_path, 'train', copy_image_to_split=True)
        save_labels_to_txt(image_path, labels_org_sort, target_image_path, 'all_org')
        save_word_image_label(image_path, words, target_word_image_path, 'train')

    for i, line in enumerate(test_samples):
        image_path, yolo_labels, labels_org_sort, words = get_path_and_labels(line, source_image_path)
        save_labels_to_txt(image_path, yolo_labels, target_image_path, 'test', copy_image_to_split=True)
        save_labels_to_txt(image_path, labels_org_sort, target_image_path, 'all_org')
        save_word_image_label(image_path, words, target_word_image_path, 'test')

    for i, line in enumerate(validation_samples):
        image_path, yolo_labels, labels_org_sort, words = get_path_and_labels(line, source_image_path)
        save_labels_to_txt(image_path, yolo_labels, target_image_path, 'val', copy_image_to_split=True)
        save_labels_to_txt(image_path, labels_org_sort, target_image_path, 'all_org')
        save_word_image_label(image_path, words, target_word_image_path, 'val')

    shutil.rmtree(source_image_path)

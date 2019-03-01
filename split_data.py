"""Split the data/dataset folder:
    c_0/
        img1.jpg
        img2.jpg
    c_1/
        img1.jpg
        img2.jpg

Into the following format:
    train/
        c_0/
            img1.jpg
            img2.jpg
         c_1/
            img1.jpg
            img2.jpg
    val/
        c_0/
            img1.jpg
            img2.jpg
         c_1/
            img1.jpg
            img2.jpg
"""
import pathlib
import random
import shutil
from os import path


train_path = "data/dataset"


def list_dirs(directory):
    """Returns all directories in a train directory
    """
    return [folder for folder in pathlib.Path(directory).iterdir()
            if folder.is_dir()]


def list_files(directory):
    """Returns all the files in a directory
    """
    return [file for file in pathlib.Path(directory).iterdir()
            if file.is_file() and not file.name.startswith('.')]


def split_ratio(input_data, seed, ratio=(0.8, 0.2)):
    assert sum(ratio) == 1
    for cls_folder in list_dirs(input_data):
        split_cls_directory(cls_folder, ratio, seed)


def split_cls_directory(directory, ratio, seed):
    random.seed(seed)
    files = list_files(directory)
    random.shuffle(files)

    split_train = int(ratio[0] * len(files))

    data = split_files(files, split_train)
    copy_files(data, directory)


def split_files(files, split_train):
    files_train = files[:split_train]
    files_val = files[split_train:]

    data = [(files_train, 'data/train'), (files_val, 'data/val')]
    return data


def copy_files(data, directory):
    """Copy the files to the respective folders"""
    class_name = path.split(directory)[1]
    for (files, data_type) in data:
        full_path = path.join(data_type, class_name)
        pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(file, full_path)


x = split_ratio(train_path, ratio=(0.8, 0.2), seed=42)
print(x)

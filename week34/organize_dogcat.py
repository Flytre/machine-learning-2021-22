from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

dataset_home = 'dogcat_labelled/'
sub_directories = ['train/', 'test/']
for cd in sub_directories:
    labels = ['dogs/', 'cats/']
    for label in labels:
        new_dir = dataset_home + cd + label
        makedirs(new_dir, exist_ok=True)

seed(1)
val_ratio = 0.25

src_directory = 'train/'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/' + file
        copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/' + file
        copyfile(src, dst)

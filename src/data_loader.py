import os
import random
import numpy as np
from scipy import misc

import utils

DATA_DIR = "../data/"
NUM_CLASS = 100

class DataLoader:

    def __init__(self, split):
        im_list_path = os.path.join(DATA_DIR, "{}.txt".format(split))
        self.im_list, self.labels = open_im_list(im_list_path)

        self.image_dir = os.path.join(DATA_DIR, "images")

    def random_im(self):
        return random.choice(self.im_list)

    def get_image(self, im):
        path = os.path.join(self.image_dir, im)
        img = misc.imread(path)
        return img

    def get_label(self, im):
        c = self.labels[im]
        label = np.zeros(NUM_CLASS)
        label[c] = 1
        return label

def open_im_list(list_path):
    im_list = []
    labels = {}
    for line in open(list_path, 'r'):
        line = line.rstrip()
        im = line.split()[0]
        c = line.split()[1]

        im_list.append(im)
        labels[im] = int(c)
    return im_list, labels


if __name__ == "__main__":
    split = "train"
    data_loader = DataLoader(split)

    means = []
    for n, im in enumerate(data_loader.im_list):
        print n, im
        img = data_loader.get_image(im)
        mean_pixel = np.mean(img, axis=(0,1))
        means.append(mean_pixel)

    means = np.stack(means, axis=0)
    mean = np.mean(means, axis=0)
    print mean



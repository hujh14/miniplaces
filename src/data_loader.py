import os
import random
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2
#import xmltodict

import utils

DATA_DIR = "../data/"
NUM_CLASS = 100

class DataLoader:

    def __init__(self, split):
        im_list_path = os.path.join(DATA_DIR, "{}.txt".format(split))
        self.im_list, self.labels = open_im_list(im_list_path)

        self.image_dir = os.path.join(DATA_DIR, "images")
        self.objects_dir = os.path.join(DATA_DIR, "objects")

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
    def get_object_mask(self, im):
        path = os.path.join(self.objects_dir, im.replace('.jpg', '.xml'))
        object_mask = load_object_mask(path)
        return object_mask


def open_im_list(list_path):
    im_list = []
    labels = {}
    for line in open(list_path, 'r'):
        line = line.rstrip()
        split = line.split()
        im = split[0]
        im_list.append(im)

        if len(split) == 2:
            c = line.split()[1]
            labels[im] = int(c)
    return im_list, labels

def load_object_mask(path):
    objects = load_objects(path)
    mask = np.zeros((128,128,175), dtype="uint8")
    for c in objects:
        c_mask = np.zeros((128,128), dtype='uint8')
        for pts in objects[c]:
            cnt = np.array(pts, dtype=int)
            cnt = cnt[:,np.newaxis,:]
            cv2.drawContours(c_mask, [cnt], -1, (255,255,255), -1)
        mask[:,:,c-1] = c_mask
    return mask

def load_objects(path):
    objects = {}
    with open(path, 'r') as f:
        data = f.read()
        data = "<root>" + data + "</root>"
        tree = xmltodict.parse(data)
        objs = tree["root"]["objects"]
        for obj in objs:
            c = int(obj["class"])
            polygon = obj["polygon"]
            points = []
            for pt in polygon["pt"]:
                x = pt["x"]
                y = pt["y"]
                points.append((x,y))

            if c not in objects:
                objects[c] = []
            objects[c].append(points)
    return objects

if __name__ == "__main__":
    split = "train"
    data_loader = DataLoader(split)

    means = []
    for n, im in enumerate(data_loader.im_list):
        print n, im
        img = data_loader.get_image(im)
        data_loader.get_label(im)
        data_loader.get_object_mask(im)
        mean_pixel = np.mean(img, axis=(0,1))
        means.append(mean_pixel)

    means = np.stack(means, axis=0)
    mean = np.mean(means, axis=0)
    print mean



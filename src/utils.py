import os
import numpy as np

PATH = os.path.dirname(__file__)

def get_categories():
    categories = {}
    with open(os.path.join(PATH, "../data/categories.txt"), 'r') as f:
        for line in f.readlines():
            split = line.split()
            category = split[0]
            c = split[1]
        return categories
categories = get_categories()

def get_latest_checkpoint(checkpoint_dir):
    # weights.00-1.52.hdf5
    latest_i = -1
    latest_fn = ""
    for fn in os.listdir(checkpoint_dir):
        split0 = fn.split('-')[0]
        i = int(split0.split('.')[1])
        if i > latest_i:
            latest_i = i
            latest_fn = fn

    if latest_i == -1:
        raise Exception("No checkpoint found.")
    return os.path.join(checkpoint_dir, latest_fn), latest_i+1


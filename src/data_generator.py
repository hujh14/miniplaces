import random
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

import utils
import image_utils
from data_loader import DataLoader

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def DataGenerator(data_loader, batch_size=8, augment=False):
    while True:
        imgs = []
        labels = []
        while len(imgs) < batch_size:
            im = data_loader.random_im()
            img = data_loader.get_image(im)
            label = data_loader.get_label(im)

            img = image_utils.preprocess_image(img)
            # Data augmentations
            if augment:
                img = add_augmentations(img)

            imgs.append(img)
            labels.append(label)

        data = np.stack(imgs, axis=0)
        label = np.stack(labels, axis=0)
        yield (data, label)
def add_augmentations(img):
    if random.randint(0,1) == 0:
        # Flip image lr
        img = np.flip(img, axis=0)
    if random.randint(0,1) == 0:
        # Flip image up
        img = np.flip(img, axis=1)
    return img


if __name__ == "__main__":
    split = "train"
    data_loader = DataLoader(split)
    generator = DataGenerator(data_loader)

    data, label = generator.next()
    print data.shape
    print label.shape

    for img in data:
        plt.imshow(img)
        plt.show()

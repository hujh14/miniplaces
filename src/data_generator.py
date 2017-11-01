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
def DataGenerator(data_loader, batch_size=32, flip=False, crop=True):
    while True:
        imgs = []
        labels = []
        while len(imgs) < batch_size:
            im = data_loader.random_im()
            img = data_loader.get_image(im)
            label = data_loader.get_label(im)

            img = image_utils.preprocess_image(img)

            # Data augmentations
            if flip:
                if random.randint(0,1) == 0:
                    # Flip image lr
                    img = np.flip(img, axis=1)
            imgs.append(img)
            labels.append(label)

            if crop:
                img = crop_split(img)
                label = [label, label, label, label]
                imgs.extend(img)
                labels.extend(label)

        data = np.stack(imgs[:batch_size], axis=0)
        label = np.stack(labels[:batch_size], axis=0)
        yield (data, label)

def crop_split(img):
    h,w = img.shape[:2]
    f = 0.875
    fh = int(f*h)
    fw = int(f*w)
    tl = img[:fh,:fw,:]
    tr = img[h-fh:,:fw,:]
    bl = img[:fh,w-fw:,:]
    br = img[h-fh:,w-fw:,:]
    imgs = [tl,tr,bl,br]
    imgs = [misc.imresize(img, (h,w)) for img in imgs]
    return imgs

if __name__ == "__main__":
    split = "train"
    data_loader = DataLoader(split)
    generator = DataGenerator(data_loader, flip=True, crop=True)

    data, label = generator.next()
    print data.shape
    print label.shape

    for img in data:
        plt.imshow(img)
        plt.show()

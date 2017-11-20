import numpy as np
from scipy import misc

DATA_MEAN = np.array([ 115.85304361,  111.24224437,  103.18997383])
INPUT_SHAPE = (224, 224)

def preprocess_image(img):
    """Preprocess an image as input."""
    f_img = img.astype('float32')
    if f_img.ndim != 3:
        f_img = np.stack((f_img,f_img,f_img), axis=2)

    centered_img = f_img - DATA_MEAN
    resized_img = misc.imresize(centered_img, INPUT_SHAPE)
    return resized_img

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
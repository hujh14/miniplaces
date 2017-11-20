import numpy as np
from scipy import misc

DATA_MEAN = np.array([ 115.85304361,  111.24224437,  103.18997383])
INPUT_SHAPE = (224, 224)
INPUT_SHAPE = (128, 128)

def preprocess_image(img):
    """Preprocess an image as input."""
    f_img = img.astype('float32')
    if f_img.ndim != 3:
        f_img = np.stack((f_img,f_img,f_img), axis=2)

    centered_img = f_img - DATA_MEAN
    resized_img = misc.imresize(centered_img, INPUT_SHAPE)
    return resized_img

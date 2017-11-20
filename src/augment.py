import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from data_loader import DataLoader

def get_augmentations(img):
    flipped = flip_lr(img)
    cropped = random_crop(img)
    shifted = shift(img)
    rotated = rotate(img)
    sheared = shear(img)
    brighter = brightness(img)
    contrasted = contrast(img)
    dropped = dropout(img)

    imgs = []
    imgs.append(flipped)
    imgs.append(cropped)
    imgs.append(shifted)
    imgs.append(rotated)
    imgs.append(sheared)
    imgs.append(brighter)
    imgs.append(contrasted)
    imgs.append(dropped)
    return imgs


def flip_lr(img):
    if random.randint(0,1) == 0:
        return np.flip(img, axis=1)
    return img

def random_crop(img, frac=0.875):
    h,w = img.shape[:2]
    sh = random.randint(0,(1-frac)*h)
    sw = random.randint(0,(1-frac)*w)
    eh = min(h,sh + int(frac*h))
    ew = min(w,sw + int(frac*w))

    new_img = img[sh:eh,sw:ew]
    new_img = misc.imresize(new_img, (h,w))
    return new_img

def shift(img, frac=0.875):
    h,w = img.shape[:2]
    delta = (1-frac)*h

    dx = random.randint(delta/2,delta)
    dy = random.randint(delta/2,delta)
    dx = random.choice([dx, -dx])
    dy = random.choice([dy, -dy])
    M = np.float32([[1,0,dx],[0,1,dy]])
    new_img = cv2.warpAffine(img,M,(w,h))
    return new_img

def rotate(img):
    h,w = img.shape[:2]
    angle = random.randint(20,90)
    angle = random.choice([angle, -angle])
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    dst = cv2.warpAffine(img,M,(w,h))
    return dst

def shear(img):
    rows,cols,ch = img.shape
    r1 = random.randint(20,50)
    r2 = random.randint(20,50)
    r1 = random.choice([r1, -r1])
    r2 = random.choice([r2, -r2])

    p1 = [50,50]
    p2 = [200,50]
    p2_ = [200,50+r1]
    p3 = [50,200]
    p3_ = [50+r2,200]

    pts1 = np.float32([p1,p2,p3])
    pts2 = np.float32([p1,p2_,p3_])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def brightness(img):
    delta = random.choice([-40,40])
    new_img = img.astype(int)
    new_img += delta
    new_img = np.array(np.clip(new_img,0,255), dtype='uint8')
    return new_img

def contrast(img):
    bgr = img[:,:,::-1]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(float)
    hsv[:,:,2] = 1.5*(hsv[:,:,2]-128) + 128
    hsv = np.array(np.clip(hsv,0,255), dtype='uint8')
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = bgr[:,:,::-1]
    return rgb

def dropout(img, rate=0.2):
    h,w = img.shape[:2]
    mask = np.random.random_sample((h,w)) < rate
    colors = np.array(np.random.random_sample((h,w,3))*255, dtype="uint8")
    new_img = img.copy()
    new_img[mask] = colors[mask]
    return new_img

if __name__ == "__main__":
    split = "train"
    data_loader = DataLoader(split)
    im = data_loader.random_im()
    img = data_loader.get_image(im)
    plt.imshow(img)
    plt.show()

    imgs = get_augmentations(img)
    for img in imgs:
        plt.imshow(img)
        plt.show()



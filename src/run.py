import os
from os import environ, makedirs
import sys
import argparse
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from data_loader import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help="Checkpoint to run")
    parser.add_argument('-s', '--split', type=str, required=True, help="Split to run on")
    parser.add_argument('-o', '--output_path', type=str, required=True, help="Output path")
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    data_loader = DataLoader(args.split)

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)

        network = Network(checkpoint=args.checkpoint)

        for im in data_loader.im_list:
            img = data_loader.get_image(im)
            predictions = network.predict(img)

            # Write output
            output = "{} {}".format(im, " ".join(predictions))
            print output
            # with open(args.output_path, 'w') as f:
            #     f.write(output)




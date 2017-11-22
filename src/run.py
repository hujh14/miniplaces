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
from network import Network


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, help="Checkpoint to run")
    parser.add_argument('-s', '--split', type=str, required=True, help="Split to run on")
    parser.add_argument('-o', '--output_path', type=str, default="output.txt", help="Output path")
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    data_loader = DataLoader(args.split)

    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
        print(args)

        network = Network(checkpoint=args.checkpoint)
        open(args.output_path, 'w').close()

        top1 = 0
        top5 = 0
        for n, im in enumerate(data_loader.im_list):
            img = data_loader.get_image(im)

            predictions = network.predict(img)
            # Write output
            output = "{} {}".format(im, " ".join([str(c) for c in predictions]))

            print output
            with open(args.output_path, 'a') as f:
                f.write(output + '\n')
            if args.split != "test":
                ans = data_loader.labels[im]           
                if ans == predictions[0]:
                    top1 += 1
                if ans in predictions:
                    top5 += 1
                print output, 1.*top1/(n+1), 1.*top5/(n+1)
        if args.split != "test":
            with open(args.output_path, 'a') as f:
                top1_error = 1 - 1.*top1/(n+1)
                top5_error = 1 - 1.*top5/(n+1)
                f.write('{} {}\n'.format(top1_error, top5_error))



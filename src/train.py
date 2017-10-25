import os
import sys
import argparse
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

import utils
from data_loader import DataLoader
from data_generator import DataGenerator
from network import Network

def train(network, generator, generator_val, checkpoint_dir, initial_epoch=0):
    filename = "weights.{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5"
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss')
    callbacks_list = [checkpoint]

    print("Training...")
    network.model.fit_generator(generator, 1000, epochs=100, callbacks=callbacks_list,
             verbose=1, workers=6, use_multiprocessing=True, initial_epoch=initial_epoch,
             validation_data=generator_val, validation_steps=25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help="Name to identify this model")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--id', default="0")
    args = parser.parse_args()
    print args

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    data_loader = DataLoader("train")
    data_loader_val = DataLoader("val")
    generator = DataGenerator(data_loader, augment=args.augment)
    generator_val = DataGenerator(data_loader_val, augment=False)

    # Load checkpoint
    checkpoint_dir = "../checkpoint/{}/{}".format(args.name, args.learning_rate)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint, epoch = (None, 0)
    if args.resume:
        checkpoint, epoch = utils.get_latest_checkpoint(checkpoint_dir)

    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
        print(args)
        network = Network(lr=args.learning_rate, checkpoint=checkpoint)

        train(network, generator, generator_val, checkpoint_dir, initial_epoch=epoch)



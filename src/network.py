import os
import argparse
import numpy as np
from scipy import misc

from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Flatten
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model, load_model
import tensorflow as tf

import image_utils

class Network:

    def __init__(self, lr=1e-3, checkpoint=None):
        if checkpoint is not None:
            print "Loading from checkpoint: ", checkpoint 
            self.model = load_model(checkpoint)
        else:
            print "Building new model"
            self.model = self.build_resnet(lr)

    def build_resnet(self, lr):
        inp = Input((224,224,3))
        model = ResNet50(input_tensor=inp, weights=None, classes=100)
        # model = ResNet50(input_tensor=inp, weights='imagenet', include_top=False)
        # x = Flatten()(model.outputs[0])
        # output = Dense(100, activation='softmax')(x)
        # model = Model(inputs=inp, outputs=output)

        sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                        loss="categorical_crossentropy",
                        metrics=['accuracy'])
        return model

    def predict(self, img):
        img = image_utils.preprocess_image(img)

        input_data = img[np.newaxis, :, :, :]  # Append batch dimension for keras
        prediction = self.model.predict(input_data)[0]

        top5 = self.get_top5(prediction)
        return top5

    def get_top5(self, predictions):
        sorted_pred = np.argsort(predictions)
        return sorted_pred[::-1][:5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default='../data/images/val/00000001.jpg',
                        help='Path the input image')
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        img = misc.imread(args.input_path)

        network = Network()

        predictions = network.predict(img)
        print predictions


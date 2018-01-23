#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import imageio
import cv2
import subprocess as sp
import tensorflow as tf
import picpac
# RESNET: import these for slim version of resnet
from gallery import Gallery

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'D.db', 'Directory to put the training data.')
flags.DEFINE_float('classes', 6, '')

def visualize (path, image, maxi):
    images = []
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    images.append(bgr.astype(np.uint8))

    bgrm = np.copy(bgr)


    channels = [bgrm[:, :, 0],  # R
                bgrm[:, :, 1],  # G
                bgrm[:, :, 2]]  # B

    # color by category
    for i in range(1, FLAGS.classes):   # ignore background 0
        for ch in [0, 1, 2]:
            channels[ch][maxi == i] *= 0.5
            # we use color i-1 because color 0 is not used
            channels[ch][maxi == i] += tableau20[i-1][ch]

    images.append(np.clip(bgrm, 0, 255).astype(np.uint8))

    imageio.mimsave(path + '.gif', images, duration = 0.5)
    sp.check_call('gifsicle --colors 256 -O3 < %s.gif > %s; rm %s.gif' % (path, path, path), shell=True)
    pass


def main (_):
    picpac_config = dict(seed=2016,
                shuffle=True,
                batch=1,
                annotate='json',
                channels=1,
                perturb=False,
                loop=False,
                stratify=True,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )
    tr_stream = picpac.ImageStream(FLAGS.db, **picpac_config)
    gal = Gallery('sample', ext='.gif')
    cc = 0
    for image, label, _ in tr_stream:
        cc += 1
        visualize(gal.next(), image[0], label[0, :, :, 0])
        if cc >= FLAGS.classes-1:
            break
        pass
    gal.flush()
        
if __name__ == '__main__':
    tf.app.run()


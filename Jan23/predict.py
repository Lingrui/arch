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
# RESNET: import these for slim version of resnet
import tensorflow as tf
from gallery import Gallery

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

class Model:
    def __init__ (self, path, name='logits:0'):
        graph = tf.Graph()
        with graph.as_default():
            saver = tf.train.import_meta_graph(path + '.meta')
        if False:
            for op in graph.get_operations():
                for v in op.values():
                    print(v.name)
        inputs = graph.get_tensor_by_name("images:0")
        outputs = graph.get_tensor_by_name(name)
        self.prob = tf.nn.softmax(outputs)
        self.path = path
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs
        self.saver = saver
        self.sess = None
        pass

    def __enter__ (self):
        assert self.sess is None
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config, graph=self.graph)
        #self.sess.run(init)
        self.saver.restore(self.sess, self.path)
        return self

    def __exit__ (self, eType, eValue, eTrace):
        self.sess.close()
        self.sess = None

    def apply (self, images):
        if self.sess is None:
            raise Exception('Model.apply must be run within context manager')
        return self.sess.run(self.prob, feed_dict={self.inputs: images})
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'D_model/200000', 'Directory to put the training data.')
flags.DEFINE_string('name', 'logits:0', '')
#flags.DEFINE_float('cth', 0.5, '')
#flags.DEFINE_integer('stride', 0, '')
flags.DEFINE_float('rx', 0.2, '')
flags.DEFINE_float('classes', 6, '')

def visualize (path, image, prob):
    images = []
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    images.append(bgr.astype(np.uint8))

    bgrm = np.copy(bgr)

    maxi = np.argmax(prob, axis=2)
    print(maxi.shape)
    assert (maxi >= 0).all()
    assert (maxi < FLAGS.classes).all()

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

DS = [('test.out', 'test'),
		#('test2.out', 'test2')]
   	  ]

def main (_):
    cc = 0
    with Model(FLAGS.model, name=FLAGS.name) as model:
		for out, inp in DS:
			gal = Gallery(out, ext='.gif')
			for root, dirs, files in os.walk(inp, topdown=False):
				for path in files:
					image = cv2.imread(os.path.join(root, path), cv2.IMREAD_GRAYSCALE)
					image = image.astype(np.float32)
					image = cv2.resize(image, None, fx=FLAGS.rx, fy=FLAGS.rx)

					prob = model.apply(np.expand_dims(np.expand_dims(image, axis=0), axis=3))[0]
					visualize(gal.next(), image, prob)
			gal.flush()
    pass

if __name__ == '__main__':
    tf.app.run()


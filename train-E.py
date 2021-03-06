#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
from tqdm import tqdm
from skimage import measure
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import picpac
import nets
from gallery import Gallery

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'E.db', '')
flags.DEFINE_integer('classes', 7, 'E has 7 classes')
flags.DEFINE_string('mixin', None, '')
flags.DEFINE_string('val', None, '')
flags.DEFINE_string('net', 'resnet_v1_50', '')
flags.DEFINE_string('opt', 'adam', '')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_bool('decay', True, '')
flags.DEFINE_float('decay_rate', 0.9, '')
flags.DEFINE_float('decay_steps', 10000, '')
flags.DEFINE_float('momentum', 0.99, 'when opt==mom')
flags.DEFINE_string('model', 'E_model', 'Directory to put the training data.')
flags.DEFINE_string('resume', None, '')
flags.DEFINE_integer('max_steps', 200000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('val_epochs', 200, '')  ## print evaluation
flags.DEFINE_integer('ckpt_epochs', 200, '')
flags.DEFINE_string('log', None, 'tensorboard')
flags.DEFINE_integer('max_summary_images', 20, '')
flags.DEFINE_integer('channels', 1, 'grayscale')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_float('pos_weight', None, '')
flags.DEFINE_string('val_plot', None, '')
flags.DEFINE_integer('max_to_keep', 1000, '')
flags.DEFINE_float('contour_th', 0.5, '')

def fcn_loss (logits, labels):
    # to HWC
    logits = tf.reshape(logits, (-1, FLAGS.classes)) # <---------------------!!!!!
    labels = tf.reshape(labels, (-1,))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int32(labels))
    if FLAGS.pos_weight:
        POS_W = tf.pow(tf.constant(FLAGS.pos_weight, dtype=tf.float32),
                       labels)
        xe = tf.multiply(xe, POS_W)
    loss = tf.reduce_mean(xe, name='xe')
    return loss, [loss] #, [loss, xe, norm, nz_all, nz_dim]

def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.db and os.path.exists(FLAGS.db)

    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    Y = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="labels")

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
                            padding='SAME'):
        logits, stride = getattr(nets, FLAGS.net)(X, num_classes=FLAGS.classes)
    loss, metrics = fcn_loss(logits, Y)
    #tf.summary.scalar("loss", loss)
    metric_names = [x.name[:-2] for x in metrics]
    for x in metrics:
        tf.summary.scalar(x.name.replace(':', '_'), x)

    rate = FLAGS.learning_rate
    if FLAGS.opt == 'adam':
        rate /= 100
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if FLAGS.decay:
        rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
        tf.summary.scalar('learning_rate', rate)
    if FLAGS.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(rate)
    elif FLAGS.opt == 'mom':
        optimizer = tf.train.MomentumOptimizer(rate, FLAGS.momentum)
    else:
        optimizer = tf.train.GradientDescentOptimizer(rate)
        pass

    train_op = optimizer.minimize(loss, global_step=global_step)
    summary_writer = None
    train_summaries = tf.constant(1)
    #val_summaries = tf.constant(1)
    if FLAGS.log:
        train_summaries = tf.summary.merge_all()
        assert not train_summaries is None
        if not train_summaries is None:
            summary_writer = tf.summary.FileWriter(FLAGS.log, tf.get_default_graph(), flush_secs=20)
        #assert train_summaries
        #val_summaries = tf.summary.merge_all(key='val_summaries')

    picpac_config = dict(seed=2016,
                shuffle=True,
                reshuffle=True,
                batch=1,
                split=1,
                split_fold=0,
                round_div=stride,
                annotate='json',
                channels=FLAGS.channels,
                stratify=True,
                pert_color1=20,
                pert_angle=180,
                pert_min_scale=0.8,
                pert_max_scale=1.2,
                pert_hflip=True,
                pert_vflip=True,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )
    if not FLAGS.mixin is None:
        picpac_config['mixin'] = FLAGS.mixin
        picpac_config['mixin_group_delta'] = 1

    tr_stream = picpac.ImageStream(FLAGS.db, perturb=True, loop=True, **picpac_config)
    val_stream = None
    if FLAGS.val:
        assert os.path.exists(FLAGS.val)
        val_stream = picpac.ImageStream(FLAGS.val, perturb=False, loop=False, **picpac_config)


    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    # check if padding have been properly setup
    # this should be ensured if all layers are added via slim
    graph = tf.get_default_graph()
    graph.finalize()
    graph_def = graph.as_graph_def()

    with tf.Session(config=config) as sess:
        sess.run(init)
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)
        step = 0
        epoch = 0
        global_start_time = time.time()
        while step < FLAGS.max_steps:
            start_time = time.time()
            avg = np.array([0] * len(metrics), dtype=np.float32)
            for _ in tqdm(range(FLAGS.epoch_steps), leave=False):
                images, labels, _ = tr_stream.next()
                feed_dict = {X: images, Y: labels}
                mm, _, summaries = sess.run([metrics, train_op, train_summaries], feed_dict=feed_dict)
                avg += np.array(mm)
                step += 1
                pass
            avg /= FLAGS.epoch_steps
            stop_time = time.time()
            txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(avg))])
            print('step %d: elapsed=%.4f time=%.4f, %s'
                    % (step, (stop_time - global_start_time), (stop_time - start_time), txt))
            if summary_writer:
                summary_writer.add_summary(summaries, step)
            epoch += 1
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, step)
                start_time = time.time()
                saver.save(sess, ckpt_path)
                stop_time = time.time()
                print('epoch %d step %d, saving to %s in %.4fs.' % (epoch, step, ckpt_path, stop_time - start_time))
            if epoch and (epoch % FLAGS.val_epochs == 0) and val_stream:
                val_stream.reset()
                avg = np.array([0] * len(metrics), dtype=np.float32)
                cc = 0
                for images, labels, _ in val_stream:
                    _, H, W, _ = images.shape
                    if FLAGS.max_size:
                        if max(H, W) > FLAGS.max_size:
                            continue
                    if FLAGS.padding == 'SAME' and FLAGS.clip:
                        images = clip(images, stride)
                        labels = clip(labels, stride)
                    feed_dict = {X: images, Y: labels}
                    #print("XXX", images.shape)
                    mm, = sess.run([metrics], feed_dict=feed_dict)
                    avg += np.array(mm)
                    cc += 1
                avg /= cc
                txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(avg))])
                print('epoch %d step %d, validation %s'
                        % (epoch, step, txt))

            pass
        pass
    if summary_writer:
        summary_writer.close()
    pass

if __name__ == '__main__':
    tf.app.run()


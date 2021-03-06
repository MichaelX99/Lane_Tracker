import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
from scipy import misc

import time


IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(119, 10, 32)]
                # 18 = bicycle

lane_colors = [(255, 0, 255)]
              # 0 = lane


class PSPNet(object):
    def __init__(self, decay=None, training=True):
        if training:
            self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        else:
            self.is_training = tf.constant(False, dtype=tf.bool, shape=[], name="is_training")

        if decay is not None:
            self.decay = decay
        else:
            self.decay = 0.

    def block(self, input, outs, sizes, strides, pad, names, rate=None, trainable=False):
        conv1 = self.compound_conv(input, outs[0], sizes[0], strides[0], names[0], trainable=trainable)

        pad =  tf.pad(conv1, paddings=np.array([[0,0], [pad, pad], [pad, pad], [0, 0]]), name=names[1])

        if rate != None:
            conv2 = self.compound_atrous_conv(pad, outs[1], sizes[1], strides[1], rate, names[2])
        else:
            conv2 = self.compound_conv(pad, outs[1], sizes[1], strides[1], names[2], trainable=trainable)

        conv3 = self.compound_conv(conv2, outs[2], sizes[2], strides[2], names[3], relu=False, trainable=trainable)

        return conv3

    def compound_atrous_conv(self, input, output, shape, stride, rate, name):
        with slim.arg_scope([slim.conv2d],
                             activation_fn=None,
                             padding='VALID',
                             biases_initializer=None):

            conv = slim.conv2d(inputs=input, num_outputs=output, kernel_size=shape, stride=stride, rate=rate, scope=name, trainable=False)
            conv = tf.layers.batch_normalization(conv, momentum=.95, epsilon=1e-5, fused=True, training=self.is_training, name=name+'_bn', trainable=False)

            conv = tf.nn.relu(conv, name=name+'_bn_relu')

            return conv

    def get_var(self, name, shape):
        return tf.get_variable(name, shape, trainable=False)

    def compound_conv(self, input, output, shape, stride, name, relu=True, padding='VALID', trainable=False):
        with slim.arg_scope([slim.conv2d],
                             activation_fn=None,
                             padding=padding,
                             biases_initializer=None):

            conv = slim.conv2d(inputs=input, num_outputs=output, kernel_size=shape, stride=stride, scope=name, trainable=trainable)
            conv = tf.layers.batch_normalization(conv, momentum=.95, epsilon=1e-5, fused=True, training=self.is_training, name=name+'_bn', trainable=trainable)


            if relu == True:
                conv = tf.nn.relu(conv, name=name+'_bn_relu')

            return conv

    def skip_connection(self, in1, in2, name):
        add = tf.add_n([in1, in2], name=name)
        add = tf.nn.relu(add, name=name+'_relu')

        return add

    def ResNet101(self, input):
        conv1_1_3x3_s2 = self.compound_conv(input, 64, 3, 2, 'conv1_1_3x3_s2', padding='SAME')

        conv1_2_3x3 = self.compound_conv(conv1_1_3x3_s2, 64, 3, 1, 'conv1_2_3x3', padding='SAME')

        conv1_3_3x3 = self.compound_conv(conv1_2_3x3, 128, 3, 1, 'conv1_3_3x3', padding='SAME')

        pool1_3x3_s2 = tf.nn.max_pool(conv1_3_3x3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1_3x3_s2')

        conv2_1_1x1_proj = self.compound_conv(pool1_3x3_s2, 256, 1, 1, 'conv2_1_1x1_proj', relu=False)

        ###################################

        outs = [64, 64, 256]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 1
        names = ['conv2_1_1x1_reduce', 'padding1', 'conv2_1_3x3', 'conv2_1_1x1_increase']
        conv2_1_1x1_increase = self.block(pool1_3x3_s2, outs, sizes, strides, pad, names)

        #####################################

        conv2_1 = self.skip_connection(conv2_1_1x1_proj, conv2_1_1x1_increase, 'conv2_1')

        outs = [64, 64, 256]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 1
        names = ['conv2_2_1x1_reduce', 'padding2', 'conv2_2_3x3', 'conv2_2_1x1_increase']
        conv2_2_1x1_increase = self.block(conv2_1, outs, sizes, strides, pad, names)

        ####################################

        conv2_2 = self.skip_connection(conv2_1, conv2_2_1x1_increase, 'conv2_2')

        outs = [64, 64, 256]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 1
        names = ['conv2_3_1x1_reduce', 'padding3', 'conv2_3_3x3', 'conv2_3_1x1_increase']
        conv2_3_1x1_increase = self.block(conv2_2, outs, sizes, strides, pad, names)

        ########################################

        conv2_3 = self.skip_connection(conv2_2, conv2_3_1x1_increase, 'conv2_3')

        conv3_1_1x1_proj = self.compound_conv(conv2_3, 512, 1, 2, 'conv3_1_1x1_proj', relu=False)

        ########################################

        outs = [128, 128, 512]
        sizes = [1, 3, 1]
        strides = [2, 1, 1]
        pad = 1
        names = ['conv3_1_1x1_reduce', 'padding4', 'conv3_1_3x3', 'conv3_1_1x1_increase']
        conv3_1_1x1_increase = self.block(conv2_3, outs, sizes, strides, pad, names)

        ###########################################

        conv3_1 = self.skip_connection(conv3_1_1x1_proj, conv3_1_1x1_increase, 'conv3_1')

        outs = [128, 128, 512]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 1
        names = ['conv3_2_1x1_reduce', 'padding5', 'conv3_2_3x3', 'conv3_2_1x1_increase']
        conv3_2_1x1_increase = self.block(conv3_1, outs, sizes, strides, pad, names)

        ##############################################

        conv3_2 = self.skip_connection(conv3_1, conv3_2_1x1_increase, 'conv3_2')

        outs = [128, 128, 512]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 1
        names = ['conv3_3_1x1_reduce', 'padding6', 'conv3_3_3x3', 'conv3_3_1x1_increase']
        conv3_3_1x1_increase = self.block(conv3_2, outs, sizes, strides, pad, names)

        #############################################

        conv3_3 = self.skip_connection(conv3_2, conv3_3_1x1_increase, 'conv3_3')

        outs = [128, 128, 512]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 1
        names = ['conv3_4_1x1_reduce', 'padding7', 'conv3_4_3x3', 'conv3_4_1x1_increase']
        conv3_4_1x1_increase = self.block(conv3_3, outs, sizes, strides, pad, names)

        ##############################################

        conv3_4 = self.skip_connection(conv3_3, conv3_4_1x1_increase, 'conv3_4')

        conv4_1_1x1_proj = self.compound_conv(conv3_4, 1024, 1, 1, 'conv4_1_1x1_proj', relu=False)

        ###############################################

        outs = [256, 256, 1024]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 2
        names = ['conv4_1_1x1_reduce', 'padding8', 'conv4_1_3x3', 'conv4_1_1x1_increase']
        conv4_1_1x1_increase = self.block(conv3_4, outs, sizes, strides, pad, names, rate=2)

        ##################################################

        conv4_names = [['conv4_2_1x1_reduce', 'padding9', 'conv4_2_3x3', 'conv4_2_1x1_increase'],
                       ['conv4_3_1x1_reduce', 'padding10', 'conv4_3_3x3', 'conv4_3_1x1_increase'],
                       ['conv4_4_1x1_reduce', 'padding11', 'conv4_4_3x3', 'conv4_4_1x1_increase'],
                       ['conv4_5_1x1_reduce', 'padding12', 'conv4_5_3x3', 'conv4_5_1x1_increase'],
                       ['conv4_6_1x1_reduce', 'padding13', 'conv4_6_3x3', 'conv4_6_1x1_increase'],
                       ['conv4_7_1x1_reduce', 'padding14', 'conv4_7_3x3', 'conv4_7_1x1_increase'],
                       ['conv4_8_1x1_reduce', 'padding15', 'conv4_8_3x3', 'conv4_8_1x1_increase'],
                       ['conv4_9_1x1_reduce', 'padding16', 'conv4_9_3x3', 'conv4_9_1x1_increase'],
                       ['conv4_10_1x1_reduce', 'padding17', 'conv4_10_3x3', 'conv4_10_1x1_increase'],
                       ['conv4_11_1x1_reduce', 'padding18', 'conv4_11_3x3', 'conv4_11_1x1_increase'],
                       ['conv4_12_1x1_reduce', 'padding19', 'conv4_12_3x3', 'conv4_12_1x1_increase'],
                       ['conv4_13_1x1_reduce', 'padding20', 'conv4_13_3x3', 'conv4_13_1x1_increase'],
                       ['conv4_14_1x1_reduce', 'padding21', 'conv4_14_3x3', 'conv4_14_1x1_increase'],
                       ['conv4_15_1x1_reduce', 'padding22', 'conv4_15_3x3', 'conv4_15_1x1_increase'],
                       ['conv4_16_1x1_reduce', 'padding23', 'conv4_16_3x3', 'conv4_16_1x1_increase'],
                       ['conv4_17_1x1_reduce', 'padding24', 'conv4_17_3x3', 'conv4_17_1x1_increase'],
                       ['conv4_18_1x1_reduce', 'padding25', 'conv4_18_3x3', 'conv4_18_1x1_increase'],
                       ['conv4_19_1x1_reduce', 'padding26', 'conv4_19_3x3', 'conv4_19_1x1_increase'],
                       ['conv4_20_1x1_reduce', 'padding27', 'conv4_20_3x3', 'conv4_20_1x1_increase'],
                       ['conv4_21_1x1_reduce', 'padding28', 'conv4_21_3x3', 'conv4_21_1x1_increase'],
                       ['conv4_22_1x1_reduce', 'padding29', 'conv4_22_3x3', 'conv4_22_1x1_increase'],
                       ['conv4_23_1x1_reduce', 'padding30', 'conv4_23_3x3', 'conv4_23_1x1_increase']]

        outs = [256, 256, 1024]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 2

        conv4_i = conv4_1_1x1_proj
        conv4_i_1x1_increase = conv4_1_1x1_increase

        conv4_i_outputs = []
        conv4_i_1x1_increase_outputs = []
        for name, i in zip(conv4_names, range(len(conv4_names))):
            i += 1
            conv4_i = self.skip_connection(conv4_i, conv4_i_1x1_increase, 'conv4_'+str(i))
            conv4_i_outputs.append(conv4_i)

            conv4_i_1x1_increase = self.block(conv4_i, outs, sizes, strides, pad, name, rate=2)
            conv4_i_1x1_increase_outputs.append(conv4_i_1x1_increase)

        ##############################################################

        conv4_22 = conv4_i_outputs[-1]
        conv4_23_1x1_increase = conv4_i_1x1_increase_outputs[-1]

        ###################################################################

        conv4_23 = self.skip_connection(conv4_22, conv4_23_1x1_increase, 'conv4_23')

        conv5_1_1x1_proj = self.compound_conv(conv4_23, 2048, 1, 1, 'conv5_1_1x1_proj', relu=False)

        return conv4_23, conv5_1_1x1_proj

    def Segmentation(self, conv4_23, conv5_1_1x1_proj, trainable=False, num_classes=19):
        outs = [512, 512, 2048]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 4
        names = ['conv5_1_1x1_reduce', 'padding31', 'conv5_1_3x3', 'conv5_1_1x1_increase']
        conv5_1_1x1_increase = self.block(conv4_23, outs, sizes, strides, pad, names, rate=4, trainable=trainable)

        ######################################################################

        conv5_1 = self.skip_connection(conv5_1_1x1_proj, conv5_1_1x1_increase, 'conv5_1')

        outs = [512, 512, 2048]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 4
        names = ['conv5_2_1x1_reduce', 'padding32', 'conv5_2_3x3', 'conv5_2_1x1_increase']
        conv5_2_1x1_increase = self.block(conv5_1, outs, sizes, strides, pad, names, rate=4, trainable=trainable)

        ##################################################################

        conv5_2 = self.skip_connection(conv5_1, conv5_2_1x1_increase, 'conv5_2')

        outs = [512, 512, 2048]
        sizes = [1, 3, 1]
        strides = [1, 1, 1]
        pad = 4
        names = ['conv5_3_1x1_reduce', 'padding33', 'conv5_3_3x3', 'conv5_3_1x1_increase']
        conv5_3_1x1_increase = self.block(conv5_2, outs, sizes, strides, pad, names, rate=4, trainable=trainable)

        ##################################################################

        conv5_3 = self.skip_connection(conv5_2, conv5_3_1x1_increase, 'conv5_3')
        shape = tf.shape(conv5_3)[1:3]

        conv5_3_pool1 = tf.nn.avg_pool(conv5_3, ksize=[1,90,90,1], strides=[1,90,90,1], padding='VALID', name='conv5_3_pool1')
        conv5_3_pool1_conv = self.compound_conv(conv5_3_pool1, 512, 1, 1, 'conv5_3_pool1_conv', trainable=trainable)
        conv5_3_pool1_interp = tf.image.resize_bilinear(conv5_3_pool1_conv, size=shape, align_corners=True, name='conv5_3_pool1_interp')

        ######################################################################

        conv5_3_pool2 = tf.nn.avg_pool(conv5_3, ksize=[1,45,45,1], strides=[1,45,45,1], padding='VALID', name='conv5_3_pool2')
        conv5_3_pool2_conv = self.compound_conv(conv5_3_pool2, 512, 1, 1, 'conv5_3_pool2_conv', trainable=trainable)
        conv5_3_pool2_interp = tf.image.resize_bilinear(conv5_3_pool2_conv, size=shape, align_corners=True, name='conv5_3_pool2_interp')

        ################################################################

        conv5_3_pool3 = tf.nn.avg_pool(conv5_3, ksize=[1,30,30,1], strides=[1,30,30,1], padding='VALID', name='conv5_3_pool3')
        conv5_3_pool3_conv = self.compound_conv(conv5_3_pool3, 512, 1, 1, 'conv5_3_pool3_conv', trainable=trainable)
        conv5_3_pool3_interp = tf.image.resize_bilinear(conv5_3_pool3_conv, size=shape, align_corners=True, name='conv5_3_pool3_interp')

        ######################################################################

        conv5_3_pool6 = tf.nn.avg_pool(conv5_3, ksize=[1,15,15,1], strides=[1,15,15,1], padding='VALID', name='conv5_3_pool6')
        conv5_3_pool6_conv = self.compound_conv(conv5_3_pool6, 512, 1, 1, 'conv5_3_pool6_conv', trainable=trainable)
        conv5_3_pool6_interp = tf.image.resize_bilinear(conv5_3_pool6_conv, size=shape, align_corners=True, name='conv5_3_pool6_interp')

        ######################################################################

        conv5_3_concat = tf.concat(axis=-1, values=[conv5_3, conv5_3_pool6_interp, conv5_3_pool3_interp, conv5_3_pool2_interp, conv5_3_pool1_interp], name='conv5_3_concat')

        conv5_4 = self.compound_conv(conv5_3_concat, 512, 3, 1, 'conv5_4', padding='SAME', trainable=trainable)

        with slim.arg_scope([slim.conv2d],
                             activation_fn=None,
                             padding='VALID'):

            conv6 = slim.conv2d(conv5_4, num_classes, [1, 1], [1, 1], scope='conv6', trainable=trainable)

        return conv6

    def inference(self, input, lane=True):

        conv4_23, conv5_1_1x1_proj = self.ResNet101(input)

        if lane:
            with tf.variable_scope("PSP"):
                psp_conv6 = self.Segmentation(conv4_23, conv5_1_1x1_proj)

            with tf.variable_scope("Lane"):
                lane_conv6 = self.Segmentation(conv4_23, conv5_1_1x1_proj, trainable=True, num_classes=2)

            return lane_conv6, psp_conv6

        else:
            psp_conv6 = self.Segmentation(conv4_23, conv5_1_1x1_proj)

            return psp_conv6

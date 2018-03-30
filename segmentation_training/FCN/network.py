import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def extract_VGG_weights(sess, vgg_path):
    #https://github.com/asimonov/CarND3-P2-FCN-Semantic-Segmentation/blob/master/main.py
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    filtered_variables = [op for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name == 'VariableV2']

    var_values = {}
    for var in filtered_variables:
        name = var.name
        tensor = tf.get_default_graph().get_tensor_by_name(name + ':0')
        value = sess.run(tensor)
        name = name.replace('filter', 'weights')
        if name[:4] == "conv":
            name = name[:5] + '/' + name
        var_values[name] = value

    return var_values

def extract_Cityscapes_weights(saved_path):
    Cityscapes_variables = {}
    Cityscapes_graph = tf.Graph()
    with Cityscapes_graph.as_default():
        seg_graph_def = tf.GraphDef()
        with tf.gfile.GFile(saved_path, 'rb') as fid:
            serialized_graph = fid.read()
            seg_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(seg_graph_def, name='')

    with tf.Session(graph = Cityscapes_graph) as sess:
        is_training = Cityscapes_graph.get_tensor_by_name("is_training:0")
        filtered_variables = [op for op in Cityscapes_graph.get_operations() if op.op_def.name == "Const"]
        for var in filtered_variables:
            name = var.name
            if ("cond" not in name) and ("nn_last_layer" not in name) and ("BatchNorm_7" not in name) and ("logits" not in name):
                #print(var.name + ": " + var.op_def.name)
                tensor = Cityscapes_graph.get_tensor_by_name(name + ':0')
                value = sess.run(tensor, feed_dict={is_training: False})
                print(name + ": " + str(np.shape(value)))
                Cityscapes_variables[name] = value

    return Cityscapes_variables

def assign_weights(sess, weights):
    #https://github.com/asimonov/CarND3-P2-FCN-Semantic-Segmentation/blob/master/fcn8vgg16.py
    tensors = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    for temp in tensors:
        key_name = temp.name[:-2]
        if key_name in weights.keys():
            print("YES: " + key_name + ", " + str(np.shape(weights[key_name])))
            sess.run(temp.assign(weights[key_name]))
        else:
            print("NO: " + key_name)
            #sess.run(temp.assign(vgg_weights[key_name]))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    t_in = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    t_kp = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    t_4 = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    t_3 = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    t_7 = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return t_in, t_kp, t_3, t_4, t_7

def vgg15(input, keep_prob):
    #https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py
    layer1 = slim.repeat(input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    layer1 = slim.max_pool2d(layer1, [2, 2], scope='pool1')
    layer2 = slim.repeat(layer1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    layer2 = slim.max_pool2d(layer2, [2, 2], scope='pool2')
    layer3 = slim.repeat(layer2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    layer3 = slim.max_pool2d(layer3, [2, 2], scope='pool3')
    layer4 = slim.repeat(layer3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    layer4 = slim.max_pool2d(layer4, [2, 2], scope='pool4')
    layer5 = slim.repeat(layer4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    layer5 = slim.max_pool2d(layer5, [2, 2], scope='pool5')

    # Use conv2d instead of fully_connected layers.
    layer6 = slim.conv2d(layer5, 4096, [7, 7], scope='fc6')
    layer6 = slim.dropout(layer6, keep_prob, scope='dropout6')
    layer7 = slim.conv2d(layer6, 4096, [1, 1], scope='fc7')
    layer7 = slim.dropout(layer7, keep_prob, scope='dropout7')

    return layer3, layer4, layer7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, is_training):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    #with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], num_outputs = num_classes,
    #                    weights_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    #                    weights_initializer=tf.random_normal_initializer(stddev=0.01),
    #                    activation_fn=None):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], num_outputs = num_classes,
                        weights_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        activation_fn=None):
        layer7a_out = slim.conv2d(vgg_layer7_out, kernel_size=[1,1])
        layer7a_out = tf.contrib.layers.batch_norm(layer7a_out, center=True, scale=True, fused=True, is_training=is_training)

        # upsample
        layer4a_in1 = slim.conv2d_transpose(layer7a_out, kernel_size=[4,4], stride=2)
        layer4a_in1 = tf.contrib.layers.batch_norm(layer4a_in1, center=True, scale=True, fused=True, is_training=is_training)

        # make sure the shapes are the same!
        # 1x1 convolution of vgg layer 4
        layer4a_in2 = slim.conv2d(vgg_layer4_out, kernel_size=[1,1])
        layer4a_in2 = tf.contrib.layers.batch_norm(layer4a_in2, center=True, scale=True, fused=True, is_training=is_training)

        # skip connection (element-wise addition)
        layer4a_out = tf.add(layer4a_in1, layer4a_in2)
        layer4a_out = tf.contrib.layers.batch_norm(layer4a_out, center=True, scale=True, fused=True, is_training=is_training)

        # upsample
        layer3a_in1 = slim.conv2d_transpose(layer4a_out, kernel_size=[4,4], stride=2)
        layer3a_in1 = tf.contrib.layers.batch_norm(layer3a_in1, center=True, scale=True, fused=True, is_training=is_training)

        # 1x1 convolution of vgg layer 3
        layer3a_in2 = slim.conv2d(vgg_layer3_out, kernel_size=[1,1])
        layer3a_in2 = tf.contrib.layers.batch_norm(layer3a_in2, center=True, scale=True, fused=True, is_training=is_training)

        # skip connection (element-wise addition)
        layer3a_out = tf.add(layer3a_in1, layer3a_in2)
        layer3a_out = tf.contrib.layers.batch_norm(layer3a_out, center=True, scale=True, fused=True, is_training=is_training)

        # upsample
        nn_last_layer = slim.conv2d_transpose(layer3a_out, kernel_size=[16,16], stride=8, scope="nn_last_layer")
        nn_last_layer = tf.contrib.layers.batch_norm(nn_last_layer, center=True, scale=True, fused=True, is_training=is_training)

    return nn_last_layer

import os.path
import tensorflow as tf
import helper
import numpy as np
from glob import glob

import tensorflow.contrib.slim as slim

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from network import *

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))

    # define training operation
    momentum = .9
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)

    regularization_loss = tf.add_n(tf.losses.get_regularization_losses())

    total_loss = regularization_loss + cross_entropy_loss

    #train_op = optimizer.minimize(cross_entropy_loss)
    train_op = optimizer.minimize(total_loss)

    return logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, is_training):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    print("Training...\n")
    count = 0
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009, is_training: True})
            if count % 5 == 0:
                print("Loss: = {:.3f}".format(loss))
            count += 1
        print("")

def train_and_save(sess, runs_dir, image_shape, num_classes, batch_size, Pretrained_variables, epochs, save_name, data_dir, pickle_files=None):
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")
    is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

    # Create function to get batches
    if pickle_files != None:
        get_batches_fn = helper.Cityscapes_gen_batch_function(pickle_files)
    else:
        get_batches_fn = helper.KITTI_gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = vgg15(input_image, keep_prob)

    nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, is_training)

    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

    sess.run(tf.global_variables_initializer())

    assign_weights(sess, Pretrained_variables)

    train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, is_training)

    logits = tf.identity(logits, "logits")

    saver = tf.train.Saver()

    print("Saving Model")
    saver.save(sess, runs_dir + '/' + save_name + '.ckpt')
    tf.train.write_graph(sess.graph_def, runs_dir, save_name + '.pb', False)
    print("Saved model.  Now starting evaluation")


    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, is_training)

    MODEL_NAME = 'seg'
    input_graph_path = 'runs/' + save_name + '.pb'
    checkpoint_path = 'runs/' + save_name + '.ckpt'
    input_saver_def_path = ""
    input_binary = True
    output_node_names = "logits"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = 'runs/frozen_' + save_name + '.pb'
    clear_devices = True
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                      input_binary, checkpoint_path, output_node_names,
                      restore_op_name, filename_tensor_name,
                      output_frozen_graph_name, clear_devices, "")

def Cityscapes_train(image_shape, batch_size):
    num_classes = 20
    epochs = 50
    data_dir = './data'
    runs_dir = './runs'
    save_name = "city"

    train_cities = ["aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf", "erfurt", "hamburg", "hanover",
              "jena", "krefeld", "monchengladbach", "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich"]

    validation_cities = ["frankfurt", "lindau", "munster"]

    testing_cities = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]

    Cityscapes_dir = "/home/mikep/DataSets/Cityscapes"
    labels_dir = Cityscapes_dir + "/Labels/"

    city_labels = []
    city_imgs = []
    for city in train_cities:
        train_dir = labels_dir + "train/"
        img_dir = Cityscapes_dir + "/Train/"
        labels = glob(train_dir + city + "/*gtFine_color.png")
        labels.sort()
        imgs = glob(img_dir + "left_img/" + city + "/*leftImg8bit.png")
        imgs.sort()
        if len(labels) == 0 or len(imgs) == 0:
            print("error")

        for label in labels:
            city_labels.append(label)
        for img in imgs:
            city_imgs.append(img)


    for city in validation_cities:
        val_dir = labels_dir + "val/"
        img_dir = Cityscapes_dir + "/Val/"
        labels = glob(val_dir + city + "/*gtFine_color.png")
        labels.sort()
        imgs = glob(img_dir + "left_img/" + city + "/*leftImg8bit.png")
        imgs.sort()
        if len(labels) == 0 or len(imgs) == 0:
            print("error")

        for label in labels:
            city_labels.append(label)
        for img in imgs:
            city_imgs.append(img)

    for city in testing_cities:
        val_dir = labels_dir + "test/"
        img_dir = Cityscapes_dir + "/Test/"
        labels = glob(val_dir + city + "/*gtFine_color.png")
        labels.sort()
        imgs = glob(img_dir + "left_img/" + city + "/*leftImg8bit.png")
        imgs.sort()
        if len(labels) == 0 or len(imgs) == 0:
            print("error")

        for label in labels:
            city_labels.append(label)
        for img in imgs:
            city_imgs.append(img)

    save_path = data_dir + "/cityscapes/"
    pickle_files = helper.prepape_dataset(save_path, city_imgs, city_labels, num_classes, image_shape, batch_size)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    vgg_path = os.path.join(data_dir, 'vgg')

    VGG_graph = tf.Graph()
    Seg_graph = tf.Graph()

    with tf.Session(graph=VGG_graph) as sess:
        vgg_weights = extract_VGG_weights(sess, vgg_path)

    with tf.Session(graph=Seg_graph) as sess:
        train_and_save(sess, runs_dir, image_shape, num_classes, batch_size, vgg_weights, epochs, save_name, data_dir, pickle_files=pickle_files)




def KITTI_train(image_shape, saved_path, batch_size):
    Cityscapes_variables = extract_Cityscapes_weights(saved_path)

    data_dir = './data'
    runs_dir = './runs'
    save_name = "final"
    num_classes = 2
    epochs = 15
    KITTI_graph = tf.Graph()
    with tf.Session(graph = KITTI_graph) as sess:
        train_and_save(sess, runs_dir, image_shape, num_classes,batch_size,  Cityscapes_variables, epochs, save_name, data_dir=data_dir)

def run():
    image_shape = (160, 576)
    batch_size = 3

    #Cityscapes_train(image_shape, batch_size)

    saved_path = "./runs/frozen_city.pb"

    KITTI_train(image_shape, saved_path, batch_size)




if __name__ == '__main__':
    run()

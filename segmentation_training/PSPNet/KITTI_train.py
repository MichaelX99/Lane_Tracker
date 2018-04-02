import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob
import time

from PSPNet import *
#from helper import *

from Image_Reader import ImageReader

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

EPOCHS = 15
BATCH_SIZE = 2
NUM_STEPS = int(EPOCHS  * 94. // BATCH_SIZE)
DATA_DIRECTORY = "../data_road/training/"
INPUT_SIZE = (713,713)
MOMENTUM = 0.9
NUM_CLASSES = 2
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
SNAPSHOT_DIR = './KITTI_model/'


def Load_Cityscapes_Model(path):
    Cityscapes_Graph = tf.Graph()

    with tf.Session(graph=Cityscapes_Graph) as sess:
        temp = tf.placeholder(tf.float32, [None, None, None, 3])

        net_obj = PSPNet()
        conv6, conv5_4, conv5_3_pool6_conv, conv5_3_pool3_conv, conv5_3_pool2_conv, conv5_3_pool1_conv = net_obj.inference(temp, lane=False)

        init = tf.global_variables_initializer()

        sess.run(init)

        restore_var = tf.global_variables()

        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])

            loader.restore(sess, ckpt.model_checkpoint_path)
            print("Model Loaded")
        else:
            print("No Model Found")

        model_vars = {}

        for var in restore_var:
            name = var.name
            if "weights" in name or "biases" in name or "gamma" in name or "beta" in name or "moving_mean" in name or "moving_variance" in name:
                print("Retreived " + var.name)
                tensor = Cityscapes_Graph.get_tensor_by_name(name)
                model_vars[name] = tensor.eval()

        print("Finished Retreiving Model Variables\n")

    return model_vars

def Train_KITTI(model_vars):
    KITTI_Graph = tf.Graph()
    with KITTI_Graph.as_default():

        tf.set_random_seed(RANDOM_SEED)
        coord = tf.train.Coordinator()

        img_mean = IMG_MEAN

        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                data_dir=DATA_DIRECTORY,
                input_size=INPUT_SIZE,
                random_scale=True,
                random_mirror=True,
                img_mean=img_mean,
                coord=coord)

            image_batch, label_batch = reader.dequeue(BATCH_SIZE)


        net_obj = PSPNet(decay=WEIGHT_DECAY)
        conv7, conv6, conv5_4, conv5_3_pool6_conv, conv5_3_pool3_conv, conv5_3_pool2_conv, conv5_3_pool1_conv = net_obj.inference(image_batch)

        conv7 = tf.identity(conv7, "lane_segmentation")
        conv6 = tf.identity(conv6, "psp_segmentation")

        # Predictions: ignoring all predictions with labels greater or equal than n_classes
        raw_prediction = tf.reshape(conv7, [-1, NUM_CLASSES])
        label_proc = prepare_label(label_batch, tf.stack(conv7.get_shape()[1:3]), num_classes=NUM_CLASSES, one_hot=False) # [batch_size, h, w]
        raw_gt = tf.reshape(label_proc, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, NUM_CLASSES - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        #if NUM_CLASSES == 1:
        #    gt = tf.reshape(gt, [-1, NUM_CLASSES])

        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax loss.
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)


        my_label = tf.image.resize_nearest_neighbor(label_batch, tf.stack(conv7.get_shape()[1:3]))
        #hot = tf.one_hot(my_label, NUM_CLASSES)

        my_pred = tf.reshape(conv7, [-1, NUM_CLASSES])
        my_label = tf.reshape(my_label, [-1])
        hot = tf.one_hot(my_label, NUM_CLASSES)


        #loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=conv7, labels=hot)
        loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=my_pred, labels=hot)

        lane_indices = tf.where(tf.equal(my_label, 0))
        other_indices = tf.where(tf.equal(my_label, 1))

        lane_pred = tf.gather(my_pred, lane_indices)
        lane_label = tf.gather(my_label, lane_indices)
        hot_lane = tf.one_hot(lane_label, NUM_CLASSES)

        other_pred = tf.gather(my_pred, other_indices)
        other_label = tf.gather(my_label, other_indices)
        hot_other = tf.one_hot(other_label, NUM_CLASSES)

        lane_loss = tf.nn.softmax_cross_entropy_with_logits(logits=lane_pred, labels=hot_lane)
        lane_loss = tf.reduce_mean(lane_loss)

        other_loss = tf.nn.softmax_cross_entropy_with_logits(logits=other_pred, labels=hot_other)
        other_loss = tf.reduce_mean(other_loss)

        #weighting = tf.divide( other_indices.get_shape(), (my_label.get_shape()) )

        weighted_loss = lane_loss + .35 * other_loss

        #loss_op = tf.reduce_mean(loss1)
        loss_op = weighted_loss

        reg_tensors = []
        for var in tf.global_variables():
            name = var.name
            if "conv7" in name:
                reg_tensors.append(tf.nn.l2_loss(var))

        #regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        regularization_loss = tf.add_n(reg_tensors)


        #total_loss = tf.reduce_mean(loss) + regularization_loss
        #total_loss = tf.reduce_mean(loss1) + regularization_loss
        total_loss = weighted_loss + regularization_loss
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(total_loss)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=KITTI_Graph)

        sess.run(init)

        for var in tf.global_variables():
            name = var.name
            if name in model_vars:
                sess.run(var.assign(model_vars[name]))
                print("Restored " + name)
            else:
                print("Did Not Restore " + name)

        print("Finished Restoring Model Variables")


        checkpoint_path = SNAPSHOT_DIR + "model.ckpt"
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        count = 0.
        print("Starting Training")
        for i in range(NUM_STEPS):
        #for i in range(1):
            """
            x, img, label, ind, lo, pred, lab = sess.run([total_loss, raw_prediction, raw_gt, indices, loss, prediction, gt], feed_dict={net_obj.is_training: True})
            lane_count = 0
            other = 0
            for j in label:
                if j == 0:
                    lane_count += 1
                else:
                    other += 1
            print(lane_count)
            print(other)
            print("\n\n\n")

            #for j in ind:
            #    print(j)
            print(x)

            for j, k, l in zip(pred, lab, lo):
                print("pred = " + str(j) + ", label = " + str(k) + "; loss = " + str(l))


            #for i in ind:
            #    print(i)

            #print("\n\n")
            #print(l)
            """
            #_, loss = sess.run([train_op, total_loss], feed_dict={net_obj.is_training: True})
            _, loss = sess.run([train_op, loss_op], feed_dict={net_obj.is_training: True})
            if count % 5 == 0:
                print("Loss: = {:.3f}".format(loss))
            count += 1

        coord.request_stop()
        coord.join(threads)

        saver.save(sess, checkpoint_path)
        print("Finished Training, Extracting weights from graph")

        """
        KITTI_vars = {}
        for var in tf.global_variables():
            name = var.name
            KITTI_vars[name] = var.eval(session=sess)
            print("Extracted " + name)

        return KITTI_vars
        """
        return 0

def freeze_and_optimize_graph(KITTI_vars):
    Save_Graph = tf.Graph()
    with Save_Graph.as_default():

        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')

        save_obj = PSPNet(training=False)
        conv7, conv6, _, _, _, _, _ = save_obj.inference(input_image)

        conv7 = tf.identity(conv7, "lane_segmentation")
        conv6 = tf.identity(conv6, "psp_segmentation")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        save_sess = tf.Session(graph=Save_Graph, config=config)

        """
        for var in tf.global_variables():
            name = var.name
            if name in KITTI_vars:
                print("Assigning " + name)
                save_sess.run(var.assign(KITTI_vars[name]))
        """


        tf.train.write_graph(save_sess.graph_def, SNAPSHOT_DIR, 'output.pb', False)
        print("Saved model.  Now freezing")

        MODEL_NAME = 'seg'
        input_graph_path = SNAPSHOT_DIR + 'output.pb'
        checkpoint_path = SNAPSHOT_DIR + "model.ckpt"
        input_saver_def_path = ""
        input_binary = True
        output_node_names = "lane_segmentation,psp_segmentation"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = SNAPSHOT_DIR + 'frozen.pb'
        clear_devices = True
        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")


        print("Froze graph, Now optimizing")


        frozen_graph_def = tf.GraphDef()
        with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
            data = f.read()
            frozen_graph_def.ParseFromString(data)

        optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
                frozen_graph_def,
                ["input_image"], # an array of the input node(s)
                ["lane_segmentation", "psp_segmentation"], # an array of output nodes
                tf.float32.as_datatype_enum)

        # Save the optimized graph
        output_optimized_graph_name = SNAPSHOT_DIR + 'optimized.pb'
        f = tf.gfile.FastGFile(output_optimized_graph_name, "wb")
        f.write(optimized_graph_def.SerializeToString())




if __name__ == '__main__':
    path = './model'

    model_vars = Load_Cityscapes_Model(path)
    #model_vars = {}

    KITTI_vars = Train_KITTI(model_vars)
    #KITTI_vars = {}

    freeze_and_optimize_graph(KITTI_vars)

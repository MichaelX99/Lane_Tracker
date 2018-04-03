import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob
import time

from PSPNet import *
from Yellowfin import YFOptimizer

from Image_Reader import ImageReader

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

EPOCHS = 15
BATCH_SIZE = 3
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
        psp_conv6 = net_obj.inference(temp, lane=False)

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
        lane_conv6, psp_conv6 = net_obj.inference(image_batch)

        lane_conv6 = tf.identity(lane_conv6, "lane_segmentation")
        psp_conv6 = tf.identity(psp_conv6, "psp_segmentation")

        # Predictions: ignoring all predictions with labels greater or equal than n_classes
        raw_prediction = tf.reshape(lane_conv6, [-1, NUM_CLASSES])
        label_proc = tf.image.resize_nearest_neighbor(label_batch, tf.stack(lane_conv6.get_shape()[1:3])) # as labels are integer numbers, need to use NN interp.
        raw_gt = tf.reshape(label_proc, [-1])

        indices = tf.where(tf.less_equal(raw_gt, NUM_CLASSES - 1))
        raw_prediction = tf.gather(raw_prediction, indices)
        raw_gt = tf.gather(raw_gt, indices)

        raw_prediction = tf.reshape(raw_prediction, [-1, NUM_CLASSES])
        raw_gt = tf.reshape(raw_gt, [-1])


        raw_gt = tf.cast(raw_gt, tf.int32)
        temp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=raw_prediction, labels=raw_gt)
        temp_loss = tf.reduce_mean(temp_loss)

        loss_op = temp_loss


        reg_tensors = []
        for var in tf.global_variables():
            name = var.name
            if "weights" in name or "biases" in name or "beta" in name or "gamma" in name:
                reg_tensors.append(tf.nn.l2_loss(var))

        regularization_loss = tf.add_n(reg_tensors)


        total_loss = temp_loss + (WEIGHT_DECAY * regularization_loss)

        optimizer = YFOptimizer()
        #optimizer = tf.train.AdamOptimizer()
        #optimizer = tf.train.MomentumOptimizer(learning_rate=.001, momentum=.3, use_nesterov=True)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=.0001)

        train_op = optimizer.minimize(total_loss)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=KITTI_Graph)

        sess.run(init)


        for var in tf.global_variables():
            name = var.name
            if "PSP" in name:
                name = name.replace("PSP/", "")
                if name in model_vars:
                    sess.run(var.assign(model_vars[name]))
                    print("Restored " + "PSP/" + name)
                else:
                    print("Did Not Restore " + "PSP/" + name)

            elif "Lane" in name:
                name = name.replace("Lane/", "")
                if "conv6" not in name:
                    if name in model_vars:
                        sess.run(var.assign(model_vars[name]))
                        print("Restored " + "Lane/" + name)
                    else:
                        print("Did Not Restore " + "Lane/" + name)
                else:
                    print("Did Not Restore " + "Lane/" + name)

            else:
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
            _, loss = sess.run([train_op, loss_op], feed_dict={net_obj.is_training: True})
            if count % 5 == 0:
                print("Loss: = {:.3f}".format(loss))
            count += 1

        coord.request_stop()
        coord.join(threads)

        saver.save(sess, checkpoint_path)
        print("Finished Training, Extracting weights from graph")


def freeze_and_optimize_graph():
    Save_Graph = tf.Graph()
    with Save_Graph.as_default():

        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')

        save_obj = PSPNet(training=False)
        lane_conv6, psp_conv6 = save_obj.inference(input_image)

        lane_conv6 = tf.identity(lane_conv6, "lane_segmentation")
        psp_conv6 = tf.identity(psp_conv6, "psp_segmentation")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        save_sess = tf.Session(graph=Save_Graph, config=config)


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

        """
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
        """



if __name__ == '__main__':
    path = './model'

    model_vars = Load_Cityscapes_Model(path)
    #model_vars = {}

    Train_KITTI(model_vars)

    freeze_and_optimize_graph()

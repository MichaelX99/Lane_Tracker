import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob
import time

from PSPNet import *
from helper import *

from Image_Reader import ImageReader

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

EPOCHS = 15
BATCH_SIZE = 2
NUM_STEPS = int(EPOCHS  * 94. // BATCH_SIZE)
DATA_DIRECTORY = "../data_road/training/"
IGNORE_LABEL = 255
INPUT_SIZE = (713,713)
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 2
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
RESTORE_FROM = './'
SNAPSHOT_DIR = './KITTI_model/'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 50


def Load_Cityscapes_Model(path):
    Cityscapes_Graph = tf.Graph()

    with tf.Session(graph=Cityscapes_Graph) as sess:
        temp = tf.placeholder(tf.float32, [None, None, None, 3])

        net_obj = PSPNet(num_classes=19)
        conv6, conv5_4, conv5_3_pool6_conv, conv5_3_pool3_conv, conv5_3_pool2_conv, conv5_3_pool1_conv = net_obj.inference(temp, lane=False)

        init = tf.global_variables_initializer()

        sess.run(init)

        restore_var = tf.global_variables()

        ckpt = tf.train.get_checkpoint_state(model_path)
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

def compute_image_mean(data_dir):
    img_paths = glob(data_dir + "image_2/*.png")

    if len(img_paths) != 0:
        r_sum = 0.
        g_sum = 0.
        b_sum = 0.

        for path in img_paths:
            img = misc.imread(path)
            r_sum += np.sum(img[:,:,0])
            g_sum += np.sum(img[:,:,1])
            b_sum += np.sum(img[:,:,2])

        r_sum /= len(img_paths)
        g_sum /= len(img_paths)
        b_sum /= len(img_paths)

        return np.array((r_sum, g_sum, b_sum), dtype=np.float32)

    else:
        return np.array((1,1,1), dtype=np.float32)


def Train_KITTI(model_vars):
    KITTI_Graph = tf.Graph()
    with KITTI_Graph.as_default():

        tf.set_random_seed(RANDOM_SEED)
        coord = tf.train.Coordinator()

        img_mean = compute_image_mean(DATA_DIRECTORY)


        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                data_dir=DATA_DIRECTORY,
                input_size=INPUT_SIZE,
                random_scale=True,
                random_mirror=True,
                ignore_label=IGNORE_LABEL,
                img_mean=img_mean,
                coord=coord)

            image_batch, label_batch = reader.dequeue(BATCH_SIZE)


        net_obj = PSPNet(num_classes = 2, decay=WEIGHT_DECAY)
        conv7, conv6, conv5_4, conv5_3_pool6_conv, conv5_3_pool3_conv, conv5_3_pool2_conv, conv5_3_pool1_conv = net_obj.inference(image_batch)

        conv7 = tf.identity(conv7, "lane_segmentation")
        conv6 = tf.identity(conv6, "output")

        # Predictions: ignoring all predictions with labels greater or equal than n_classes
        raw_prediction = tf.reshape(conv6, [-1, NUM_CLASSES])
        label_proc = prepare_label(label_batch, tf.stack(conv6.get_shape()[1:3]), num_classes=NUM_CLASSES, one_hot=False) # [batch_size, h, w]
        raw_gt = tf.reshape(label_proc, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, NUM_CLASSES - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        #l2_losses = [WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]

        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        #reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
        total_loss = tf.reduce_mean(loss) + regularization_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer()

            train_op = optimizer.minimize(total_loss)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=KITTI_Graph)

        sess.run(init)

        for var in tf.global_variables():
        #for var in all_trainable:
            name = var.name
            if name in model_vars:
                #sess.run(var.assign(model_vars[name]))
                print("Restored " + name)
            else:
                print("Did Not Restore " + name)

        print("Finished Restoring Model Variables")

        checkpoint_path = SNAPSHOT_DIR + "model.ckpt"
        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)


        count = 0.

        print("Starting Training")
        for i in range(NUM_STEPS):
            print(i)
            #_, loss = sess.run([train_op, total_loss], feed_dict={net_obj.is_training: True})
            #if count % 5 == 0:
            #    print("Loss: = {:.3f}".format(loss))
            #count += 1

        coord.request_stop()
        coord.join(threads)

        saver.save(sess, checkpoint_path)
        print("Finished Training, Extracting weights from graph")

        KITTI_vars = {}
        for var in tf.global_variables():
            name = var.name
            #if "weights" in name or "biases" in name or "beta" in name or "gamma" in name or "moving_mean" in name or "moving_variance" in name:
            #KITTI_vars[name] = var.eval(session=sess)
            print("Extracted " + name)

    Save_Graph = tf.Graph()
    with Save_Graph.as_default():

        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')

        save_obj = PSPNet(num_classes = 2, decay=WEIGHT_DECAY, training=False)
        conv7, conv6, _, _, _, _, _ = save_obj.inference(input_image)

        conv7 = tf.identity(conv7, "lane_segmentation")
        conv6 = tf.identity(conv6, "output")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        save_sess = tf.Session(graph=Save_Graph, config=config)

        for var in tf.global_variables():
            name = var.name
            if name in KITTI_vars:
                print("Assigning " + name)
                #save_sess.run(var.assign(KITTI_vars[name]))


        tf.train.write_graph(save_sess.graph_def, SNAPSHOT_DIR, 'output.pb', False)
        print("Saved model.  Now freezing")

        MODEL_NAME = 'seg'
        input_graph_path = SNAPSHOT_DIR + 'output.pb'
        #checkpoint_path = SNAPSHOT_DIR + 'model.ckpt'
        input_saver_def_path = ""
        input_binary = True
        output_node_names = "output"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = SNAPSHOT_DIR + 'frozen.pb'
        clear_devices = True
        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")




        frozen_graph_def = tf.GraphDef()
        with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
            data = f.read()
            frozen_graph_def.ParseFromString(data)

        optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
                frozen_graph_def,
                ["input_image"], # an array of the input node(s)
                ["conv6/BiasAdd", "output"], # an array of output nodes
                tf.float32.as_datatype_enum)

        # Save the optimized graph
        output_optimized_graph_name = SNAPSHOT_DIR + 'optimized.pb'
        f = tf.gfile.FastGFile(output_optimized_graph_name, "wb")
        f.write(optimized_graph_def.SerializeToString())






if __name__ == '__main__':
    path = './model'

    #model_vars = Load_Cityscapes_Model(path)
    model_vars = {}

    Train_KITTI(model_vars)

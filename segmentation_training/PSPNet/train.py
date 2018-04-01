import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob
import time

from PSPNet import *

from Image_Reader import ImageReader

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

EPOCHS = 15
BATCH_SIZE = 2
NUM_STEPS = EPOCHS  * 94. // BATCH_SIZE
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
            if "weights" in name or "biases" in name or "gamma" in name or "beta" in name:
                print("Retreived " + var.name)
                tensor = Cityscapes_Graph.get_tensor_by_name(name)
                model_vars[name] = tensor.eval()

        print("Finished Retreiving Model Variables\n")

    return model_vars

def compute_image_mean(data_dir):
    img_paths = glob(data_dir + "image_2/*.png")

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

        net_obj = PSPNet(num_classes = 2)
        conv7, conv6, conv5_4, conv5_3_pool6_conv, conv5_3_pool3_conv, conv5_3_pool2_conv, conv5_3_pool1_conv = net_obj.inference(image_batch)

        conv7 = tf.identity(conv7, "lane_segmentation")
        conv6 = tf.identity(conv6, "output")


        all_trainable = [v for v in tf.trainable_variables()]
        #all_trainable = [v for v in tf.global_variables()]
        conv_trainable = []
        fc_w_trainable = []
        fc_b_trainable = []
        for var in all_trainable:
            name = var.name
            if "conv6" in name or "pool" in name:
                if "weights" in name:
                    fc_w_trainable.append(var)
                elif "biases" in name:
                    fc_b_trainable.append(var)
                else:
                    conv_trainable.append(var)
            else:
                conv_trainable.append(var)


        assert(len(all_trainable) == len(fc_w_trainable) + len(fc_b_trainable) + len(conv_trainable))

        # Predictions: ignoring all predictions with labels greater or equal than n_classes
        raw_prediction = tf.reshape(conv6, [-1, NUM_CLASSES])
        label_proc = prepare_label(label_batch, tf.stack(conv6.get_shape()[1:3]), num_classes=NUM_CLASSES, one_hot=False) # [batch_size, h, w]
        raw_gt = tf.reshape(label_proc, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, NUM_CLASSES - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        l2_losses = [WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

        # Using Poly learning rate policy
        base_lr = tf.constant(LEARNING_RATE)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / NUM_STEPS), POWER))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        """
        with tf.control_dependencies(update_ops):
            opt_conv = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
            opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, MOMENTUM)
            opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, MOMENTUM)

            grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
            grads_conv = grads[:len(conv_trainable)]
            grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
            grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

            train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
            train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
            train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

            train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
        """

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=KITTI_Graph)

        sess.run(init)

        for var in tf.global_variables():
        #for var in all_trainable:
            name = var.name
            if "moving" not in name:
                if name in model_vars:
                    #sess.run(var.assign(model_vars[name]))
                    print("Restored " + name)
                else:
                    print("Did Not Restore " + name)

        print("Finished Restoring Model Variables")
        """
        checkpoint_path = SNAPSHOT_DIR + "model.ckpt"

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        saver.save(sess, checkpoint_path)
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Iterate over training steps.
        for step in range(NUM_STEPS):
            start_time = time.time()

            feed_dict = {step_ph: step}
            if step % SAVE_PRED_EVERY == 0:
                loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
                save(saver, sess, checkpoint_path, step)
            else:
                loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

        coord.request_stop()
        coord.join(threads)


        tf.train.write_graph(sess.graph_def, SNAPSHOT_DIR, 'output.pb', False)
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

        """


if __name__ == '__main__':
    path = './model'

    model_vars = Load_Cityscapes_Model(path)
    #model_vars = {}

    Train_KITTI(model_vars)

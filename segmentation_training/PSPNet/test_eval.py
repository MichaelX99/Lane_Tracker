import tensorflow as tf
import numpy as np
from scipy import misc
import cv2

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

# THESE LABELS ARE NOT THE OFFICIAL CITYSCAPES LABELS AND ARE INSTEAD HELLOCHICKS OWN CONVENTION
# TODO change these back to the correct values
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
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

lane_colors = [(0, 255, 0), (0,0,0)]
              # 0 = lane

crop_size = [720, 720]

def decode_labels(mask, img_shape, num_classes):
    if num_classes == 19:
        color_table = label_colours
    elif num_classes == 2:
        color_table = lane_colors

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3))

    return pred

def load_img(img_path):
    img = misc.imread(img_path)
    img_shape = img.shape

    h = max(crop_size[0], img_shape[0])
    w = max(crop_size[1], img_shape[1])

    return img, h, w

def preprocess(img, h, w):
    # Convert RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    temp = np.ones(img.shape)
    for i in range(3):
        temp[:,:,i] = np.multiply(temp[:,:,i], IMG_MEAN[i])

    # Extract mean.
    img = np.subtract(img, temp)

    dh = img.shape[0] - h
    dw = img.shape[1] - w

    if dh != 0 or dw != 0:
        pad_img = np.zeros((h, w, 3))
        pad_img[:img.shape[0], :img.shape[1]] = img
    else:
        pad_img = img


    pad_img = [pad_img]

    return pad_img

def inference(raw_output, img, num_classes):
    img_shape = img[0].shape

    h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    pred = decode_labels(raw_output_up, img_shape, num_classes)

    return pred

frozen_graph = tf.Graph()
frozen_path = './KITTI_model/frozen.pb'

with frozen_graph.as_default():
        frozen_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_path, 'rb') as fid:
            frozen_serialized_graph = fid.read()
            frozen_graph_def.ParseFromString(frozen_serialized_graph)
            tf.import_graph_def(frozen_graph_def, name='')

        input_img = frozen_graph.get_tensor_by_name("input_image:0")
        pspnet_tensor = frozen_graph.get_tensor_by_name("psp_segmentation:0")
        lane_tensor = frozen_graph.get_tensor_by_name("lane_segmentation:0")

        img_filepath = 'lane_input.png'
        #img_filepath = 'input.png'
        img, h, w = load_img(img_filepath)
        pad_img = preprocess(img, h, w)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        frozen_sess = tf.Session(graph=frozen_graph, config=config)

        """
        pspnet_pred, lane_pred = frozen_sess.run([pspnet_tensor, lane_tensor], feed_dict={input_img: pad_img})

        for i in lane_pred:
            for j in i:
                print j
        """


        pspnet_output_op = inference(pspnet_tensor, pad_img, 19)
        lane_output_op = inference(lane_tensor, pad_img, 2)

        pspnet_pred, lane_pred = frozen_sess.run([pspnet_output_op, lane_output_op], feed_dict={input_img: pad_img})
        misc.imsave("psp_test.png", pspnet_pred[0])
        misc.imsave("lane_test.png", lane_pred[0])


"""
optimized_graph = tf.Graph()
optimized_path = './KITTI_model/optimized.pb'

with optimized_graph.as_default():
        optimized_graph_def = tf.GraphDef()
        with tf.gfile.GFile(optimized_path, 'rb') as fid:
            optimized_serialized_graph = fid.read()
            optimized_graph_def.ParseFromString(optimized_serialized_graph)
            tf.import_graph_def(optimized_graph_def, name='')


        input_img = optimized_graph.get_tensor_by_name("input_image:0")
        pspnet_tensor = optimized_graph.get_tensor_by_name("psp_segmentation:0")
        lane_tensor = optimized_graph.get_tensor_by_name("lane_segmentation:0")

        img, h, w = load_img(img_filepath)
        pad_img = preprocess(img, h, w)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        optimized_sess = tf.Session(graph=optimized_graph, config=config)

        pspnet_output_op = inference(pspnet_tensor, pad_img, 19)
        lane_output_op = inference(lane_tensor, pad_img, 2)

        pspnet_pred, lane_pred = optimized_sess.run([pspnet_output_op, lane_output_op], feed_dict={input_img: pad_img})
        misc.imsave("opt_psp_test.png", pspnet_pred[0])
        misc.imsave("opt_lane_test.png", lane_pred[0])
"""

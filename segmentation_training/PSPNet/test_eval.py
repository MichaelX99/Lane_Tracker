import tensorflow as tf
import numpy as np
from scipy import misc
import cv2

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

lane_colors = [(255, 0, 0)]
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
    #img = tf.image.decode_png(tf.read_file(img_path), channels=3)
    #img_shape = tf.shape(img)

    h, w = (np.max(crop_size[0], img_shape[0]), np.max(crop_size[1], img_shape[1]))


    return img, h, w

def preprocess(img, h, w):
    # Convert RGB to BGR
    #img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    #img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Extract mean.
    img -= IMG_MEAN

    #pad_img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
    dh = img.shape[0] - h
    dw = img.shape[1] - w
    if dh != 0 and dw != 0:
        pad_img = np.zeros((h, w, 3))
        pad_img[:img.shape[0], :img.shape[1]] = img


    pad_img = [pad_img]
    #pad_img = tf.expand_dims(pad_img, dim=0)

    return pad_img

def inference(raw_output, img):
    img_shape = img.shape

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
        pspnet_tensor = frozen_graph.get_tensor_by_name("conv6/BiasAdd:0")
        lane_tensor = frozen_graph.get_tensor_by_name("output:0")

        img_filepath = 'temp'
        img, h, w = load_img(img_filepath)
        pad_img = preprocess(img, h, w)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        frozen_sess = tf.Session(graph=frozen_graph, config=config)

        pspnet_output, lane_output = frozen_sess.run([pspnet_tensor, lane_tensor], feed_dict={input_img: pad_img})

optimized_graph = tf.Graph()
optimized_path = './KITTI_model/optimized.pb'

with optimized_graph.as_default():
        optimized_graph_def = tf.GraphDef()
        with tf.gfile.GFile(optimized_path, 'rb') as fid:
            optimized_serialized_graph = fid.read()
            optimized_graph_def.ParseFromString(optimized_serialized_graph)
            tf.import_graph_def(optimized_graph_def, name='')


        input_img = frozen_graph.get_tensor_by_name("input_image:0")
        pspnet_tensor = frozen_graph.get_tensor_by_name("conv6/BiasAdd:0")
        lane_tensor = frozen_graph.get_tensor_by_name("output:0")

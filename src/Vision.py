#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import os
import time
import tensorflow as tf
import cv2

#
# NEED TO ADD TRANSITION AND STATE VECTOR GENERATION
#
class Detector(object):
    def __init__(self):
        rospy.init_node('Lane_Detector')

        self.num_states = rospy.get_param("num_states")
        self.num_particles = rospy.get_param("num_particles")
        self.state_path = rospy.get_param("state_path")

        if !os.path.exists(self.state_path):
            self.generate_state_file()

        self.transition_path = rospy.get_param("transition_path")

        if !os.path.exists(self.transition_path):
            self.generate_transition_file()

        self.segmentation_graph = tf.Graph()
        model_path = rospy.get_param("segmentation_model_path")

        if model_path is not None:
            self.Load_Segmentation_Model(model_path)
        self.sess = tf.Session(graph=self.segmentation_graph)

        self.bridge = CvBridge()
        sub = rospy.Subscriber('/roadway/image', Image, self.Compute_Evidence)
        self.pub = rospy.Publisher('/roadway/observed_state', Float32, queue_size=1)

        rospy.spin()

    def Compute_Evidence(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg.data, "bgr8")

        Observed_State = self.Vision_System.Compute_Observed_State(img)

        self.pub.publish(Observed_State)

    def Segment_Image(self, img):
        """
        Input: A raw image of the road
        Output: A raw segmented estimation of which pixels are lane pixels
        Purpose: Perform inference on an image of a road with a lane segmentation CNN
        """
        img_shape = (160, 576)
        img = scipy.misc.imresize(img, img_shape)

        im_softmax = self.sess.run([self.segment_op], {self.keep_prob: 1.0, self.input_tensor: [img]})

        im_softmax = im_softmax[0][:, 1].reshape(img_shape[0], img_shape[1])
        segmentation = (im_softmax > 0.5).reshape(img_shape[0], img_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        #street_im = scipy.misc.toimage(img)
        #street_im.paste(mask, box=None, mask=mask)

        #return np.array(street_im)
        return mask

    def Load_Segmentation_Model(self, path):
        """
        Input: Filepath to a frozen Tensorflow segmentation model
        Output: N/A
        Purpose: Load a Tensorflow Graph into memory
        """
        with self.segmentation_graph.as_default():
            seg_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                seg_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(seg_graph_def, name='')


            self.input_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')
            self.keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
            self.logits = tf.get_default_graph().get_tensor_by_name('logits:0')

            self.segment_op = tf.nn.softmax(self.logits)

    def Blob_Detection(self, img):
        """
        Input: Raw output of the segmentation CNN
        Output: Single blob of segmented lane pixels
        Purpose: Locate the largest blob of segmented lane pixels to eliminate noise during the polynomial fitting process
        """
        return img

    def Compute_Polynomial_Fits(self, img):
        """
        Input: The most likely set of segmented lane pixels
        Output: Polynomial fits for the left and right side of the lane
        Purpose: Compute polynomial fits for the left and right side of the lane
        """
        recreated = np.zeros_like(img)

        combined_y = {}
        for y,x in zip(y_pixels, x_pixels):
            if y not in combined_y:
                combined_y[y] = [x]
            else:
                combined_y[y].append(x)

        left_pixels = []
        right_pixels = []
        for y in combined_y:
            left_most = w
            right_most = 0
            for x in combined_y[y]:
                if x < left_most:
                    left_most = x
                if x > right_most:
                    right_most = x
            left_pixels.append([y, left_most])
            right_pixels.append([y, right_most])


        lsize = len(left_pixels)
        rsize = len(right_pixels)
        l1 = int(.05 * lsize)
        l2 = int(.95 * lsize)
        r1 = int(.05 * rsize)
        r2 = int(.95 * rsize)
        left_pixels = left_pixels[l1:l2]
        right_pixels = right_pixels[r1:r2]


        for left,right in zip(left_pixels, right_pixels):
            recreated[left[0], left[1],0] = 1
            recreated[left[0], left[1],1] = 0
            recreated[left[0], left[1],2] = 1

            recreated[right[0], right[1],0] = 1
            recreated[right[0], right[1],1] = 0
            recreated[right[0], right[1],2] = 1


        left_pixels = np.array(left_pixels)
        right_pixels = np.array(right_pixels)

        left_y = left_pixels[:,0]
        left_x = left_pixels[:,1]

        right_y = right_pixels[:,0]
        right_x = right_pixels[:,1]

        left_fit = np.polyfit(left_y, left_x, 3)
        right_fit = np.polyfit(right_y, right_x, 3)

        """
        for ly, ry in zip(left_y, right_y):
            lx = int(left_fit[0]*ly**3 + left_fit[1]*ly**2 + left_fit[2]*ly + left_fit[3])
            rx = int(right_fit[0]*ry**3 + right_fit[1]*ry**2 + right_fit[2]*ry + right_fit[3])

            if lx < w:
                recreated[ly,lx,0] = 0
                recreated[ly,lx,1] = 1

            if lx < w:
                recreated[ry,rx,0] = 0
                recreated[ry,rx,1] = 1
        """

        return left_fit, right_fit


    def Compute_Observed_State(self, img):
        """
        Input: Raw image of the road
        Output: The observed state of the road
        Purpose: Take an image and compute the estimated state of the road from the given observation
        """
        output = 0.

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        segment = self.Segment_Image(img)

        blobbed = self.Blob_Detection(segment)

        left_fit, right_fit = self.Compute_Polynomial_Fits(blobbed)

        return output

    def generate_state_file(self):
        pass

    def generate_transition_file(self):
        pass

if __name__ == "__main__":
    try:
        det = Detector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start Tracker node.')

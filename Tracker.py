#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
#from Vision_System import Vision_System
import os
import time

class Tracker(object):
    """
    Class to perform detection and tracking of lanes
    """
    def __init__(self, n_particles = 100, n_states = 10000, path = None):
        rospy.init_node('Lane_Tracker')

        np.set_printoptions(precision=3) # For printing purposes

        # Make sure the number of states is even
        try:
            assert n_states % 2 == 0
        except:
            print("The number of states needs to be divisible by 2")
            exit()

        # Set up the probabalistic model
        self.n_particles = n_particles
        self.n_states = n_states
        self.mid = self.n_states // 2
        self.Initialize_Transition_Model()
        self.Initialize_Sensor_Model()
        self.Current_State = np.zeros(n_states)

        print("Done Initializing")

        self.matrix_particles = np.random.rand(n_particles, n_states)
        self.Normalize_Particles()

        start_time = time.time()
        n = 5
        for _  in range(n):
            t = time.time()
            self.matrix_particles = np.dot(self.matrix_particles, self.Transition_Model)
            self.Normalize_Particles()
            print(time.time() - t)
        print("-------------------")

        dt = time.time() - start_time
        dt /= n
        print("--- %s seconds ---" % dt)
        #self.Compute_Current_State()
        #print(self.Current_State)

        if path is not None:
            self.vision_system = Vision_System(path)
        else:
            print("Please enter a valid path to a frozen Tensorflow model")
            #exit()

        # Initialize ROS subscribers and publishers
        self.bridge = CvBridge()
        sub = rospy.Subscriber('/roadway/image', Image, self.Apply_Evidence)
        self.pub = rospy.Publisher('/roadway/estimated_state', Float32, queue_size=1)

        rospy.spin()

    def Apply_Evidence(self, msg):
        """
        Input: ROS msg of an image of the road
        Output: N/A
        Purpose: Takes an image, computes the observed state of the lane, and applies it to the current assumed state of the lane
        """
        img = self.bridge.imgmsg_to_cv2(msg.data, "bgr8")

        #Observed_State = self.Vision_System.Compute_Observed_State(img)
        #self.Reweight_Particles(Observed_State)

    def Reweight_Particles(self, Observed_State):
        """
        Input: The observed state of the lane
        Output: N/A
        Purpose: Reweight all particles given the observed evidence
        """
        Sensor_Model = self.Compute_Sensor_Model(Observed_State)
        for i in range(len(self.particles)):
            self.particles[i].Apply_Model(Sensor_Model)

    def Resample_Particles(self):
        """
        Input: N/A
        Output: N/A
        Purpose: Resample particles after being weighted according to the observed evidence
        """
        pass

    def Compute_Current_State(self):
        Current_State = np.zeros(n_states)
        for i in range(len(self.matrix_particles)):
            Current_State = np.add(Current_State, self.matrix_particles[i,:])

        s = np.sum(Current_State)
        self.Current_State = Current_State  /s

    def Normalize_Particles(self):
        for i in range(len(self.matrix_particles)):
            s = np.sum(self.matrix_particles[i,:])
            self.matrix_particles[i,:] /= s

    def Compute_Sensor_Model(self, Observed_State):
        """
        Input: Observed lane state of detected lane
        Output: Sensor Model matrix
        Purpose: Form the Sensor Model matrix in order to apply the observed evidence to the current state
        """
        Sensor_Model = self.Sensor_Model

        return Sensor_Model

    def norm(self, x, m, s=10):
        """
        Input: Point to determine the Gaussian probability density, the mean, the standard deviation
        Output: The Gaussian probability density
        Purpose: Compute the Gaussian probability density with the given parameters
        """
        V = s**2
        output = (1/np.sqrt(2 * 3.14 * V) * np.exp(-(x - m)**2/(2*V)))

        return output

    def Initialize_Sensor_Model(self):
        """
        Input: N/A
        Output: N/A
        Purpose: Initialize the Sensor Model for the particle filter
        """
        self.Sensor_Model = np.eye(self.n_states)

    def Initialize_Transition_Model(self):
        """
        Input: N/A
        Output: N/A
        Purpose: Initialize the Transition Model for the particle filter
        """
        fpath = './transition.npy'
        if os.path.isfile(fpath):
            print("Loading Transition Model")
            self.Transition_Model = np.load(fpath)
            if np.shape(self.Transition_Model) == (self.n_states, self.n_states):
                return

        print("Creating Transition Model")
        self.Transition_Model = np.zeros((self.n_states, self.n_states))

        scale = int(.1 * self.n_states)
        index = []
        for i in range(self.n_states):
            temp = []
            for j in reversed(range(scale+1)):
                ind = i - j
                if ind >= 0:
                    temp.append(ind)
            for j in range(1,scale+1,1):
                ind = i + j
                if ind <= (self.n_states - 1):
                    temp.append(ind)
            index.append(temp)

        for i, change in enumerate(index):
            for j in change:
                out = self.norm(j, i)
                self.Transition_Model[i,j] = out
            s = np.sum(self.Transition_Model[i,:])
            for j in range(self.n_states):
                self.Transition_Model[i,j] /= s

        np.save(fpath, self.Transition_Model)

if __name__ == '__main__':
    n_particles = 100
    n_states = 10000
    path = None
    #path = "/home/mikep/CarND-Semantic-Segmentation/runs/graph_def.pb"
    try:
        Track = Tracker(n_particles = n_particles, n_states = n_states, path = path)
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start Tracker node.')

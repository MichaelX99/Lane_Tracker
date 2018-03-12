import numpy as np

class Particle(object):
    def __init__(self, N):
        self.N = N
        self.Current_State = np.ones(N) * (1./self.N)

    def Apply_Model(self, Model):
        """
        Input: Matrix
        Output: N/A
        Purpose: Apply a given matrix to the current state to represent either a transition or applying evidence
        """
        self.Current_State = np.dot(Model, self.Current_State)
        s = np.sum(self.Current_State)
        self.Current_State /= s

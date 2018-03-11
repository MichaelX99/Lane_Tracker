import tensorflow as tf
import numpy as np

class Tracker(object):
    def __init__(self, N = 10000, path = None):
        self.Form_Transition(N)

        if path is not None:
            self.Load_Segmentation_Model(path)

        self.sess = tf.Session()

    def Compute_Evidence(self):
        pass

    def Apply_Transition(self):
        pass

    def Apply_Evidence(self):
        pass

    def Resample_Particles(self):
        pass

    def Compute_Curvate(self, img):
        pass

    def Segment_Image(self, img):
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
        segmentation_graph = tf.Graph()
        with segmentation_graph.as_default():
            seg_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                seg_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(seg_graph_def, name='')


            self.input_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')
            self.keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
            self.logits = tf.get_default_graph().get_tensor_by_name('logits:0')

            self.segment_op = tf.nn.softmax(self.logits)

    def Form_Transition(self, N = 10000):
        self.Transition = np.zeros((N,N))

        scale = int(.1 * N)
        index = []
        for i in range(N):
            temp = []
            for j in reversed(range(scale+1)):
                ind = i - j
                if ind >= 0:
                    temp.append(ind)
            for j in range(1,scale+1,1):
                ind = i + j
                if ind <= (N - 1):
                    temp.append(ind)
            index.append(temp)

        def norm(x, m, s=.5):
            V = s**2
            output = (1/np.sqrt(2 * 3.14 * V) * np.exp(-(x - m)**2/(2*V)))

            return output

        for i, change in enumerate(index):
            for j in change:
                out = norm(j, i)
                self.Transition[i,j] = out
            s = np.sum(self.Transition[i,:])
            for j in range(N):
                self.Transition[i,j] /= s


if __name__ == '__main__':
    N = 100
    #path = "/home/mikep/CarND-Semantic-Segmentation/runs/graph_def.pb"
    Track = Tracker(N, path=None)

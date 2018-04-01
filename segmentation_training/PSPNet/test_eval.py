import tensorflow as tf

frozen_graph = tf.Graph()
frozen_path = './KITTI_model/frozen.pb'

with frozen_graph.as_default():
        frozen_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_path, 'rb') as fid:
            frozen_serialized_graph = fid.read()
            frozen_graph_def.ParseFromString(frozen_serialized_graph)
            tf.import_graph_def(frozen_graph_def, name='')


        #for op in segmentation_graph.get_operations():
        #    print(op.name)
        print("Frozen has " + str(len(frozen_graph.get_operations())))


optimized_graph = tf.Graph()
optimized_path = './KITTI_model/optimized.pb'

with optimized_graph.as_default():
        optimized_graph_def = tf.GraphDef()
        with tf.gfile.GFile(optimized_path, 'rb') as fid:
            optimized_serialized_graph = fid.read()
            optimized_graph_def.ParseFromString(optimized_serialized_graph)
            tf.import_graph_def(optimized_graph_def, name='')


        #for op in optimized_graph.get_operations():
        #    print(op.name)
        print("Optimized has " + str(len(optimized_graph.get_operations())))


        #self.input_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')
        #self.keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        #self.logits = tf.get_default_graph().get_tensor_by_name('logits:0')

        #self.segment_op = tf.nn.softmax(self.logits)

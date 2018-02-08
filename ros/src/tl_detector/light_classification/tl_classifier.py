from styx_msgs.msg import TrafficLight
import tensorflow as tf
import scipy.misc
import cv2
import numpy as np
import time
import os
import rospy

class TLClassifier(object):
    def __init__(self, model_info=None, class_mapping=None, collect_training_data=False):
        self.model_info = model_info
        self.class_mapping = class_mapping
        self.collect_training_data = collect_training_data
        self.count = 0

        if self.model_info is not None:
            # Load the persisted model into default graph
            self.graph = tf.Graph()
            with self.graph.as_default():
                with tf.gfile.FastGFile(model_info['model_file_name'], 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')

            # Set input format
            self.input_size = (self.model_info['input_width'], self.model_info['input_height'])
            self.input_mean = self.model_info['input_mean']
            self.input_std = self.model_info['input_std']

            # Load the labels
            self.labels = [line.rstrip() for line in tf.gfile.GFile(self.model_info['labels_file_name'])]


    @staticmethod
    def save_training_img(image, state):
        if state == TrafficLight.GREEN:
            tl_color = "Green"
        elif state == TrafficLight.YELLOW:
            tl_color = "Yellow"
        elif state == TrafficLight.RED:
            tl_color = "Red"
        else:
            tl_color = "Unknown"

        directory = "gt/{}".format(tl_color)
        if not os.path.exists(directory):
            os.makedirs(directory)

        time_str = time.strftime("%Y%m%d-%H%M%S")
        file_name = directory + "/img_{}.png".format(time_str)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_name, bgr)


    """Determines the color of the traffic light in the image
    Args:
        image (cv::Mat): image containing the traffic light
        state: Current state of traffic light (ground truth only used for training)
    Returns:
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)
    """
    def get_classification(self, image, state=None):
        if self.collect_training_data and state is not None:
            # Save labeled image for training
            self.save_training_img(image, state)
            return state
        elif self.model_info is not None:

            # No clear detection
            if self.class_mapping:
                result = self.class_mapping['none']
            else:
                result = -1

            with tf.Session(graph=self.graph) as sess:
                # Preprocess
                input_image = scipy.misc.imresize(image, self.input_size, 'bilinear')
                input_image = np.squeeze((input_image.astype('Float32') - self.input_mean) / self.input_std)

                # Classifiy
                result_tensor = sess.graph.get_tensor_by_name(self.model_info['output_tensor_name'])
                predictions, = sess.run(result_tensor, {self.model_info['resized_input_tensor_name']: [input_image]})
                top_k = predictions.argsort()[-2:][::-1]
                top_class = top_k[0]

                # TODO remove debugging output
                #for node_id in top_k:
                #    print('%s (score = %.5f)' % (self.labels[node_id], predictions[node_id]))
                rospy.loginfo('%s (score = %.5f)' % (self.labels[top_class], predictions[top_class]))

                if predictions[top_class] > 0.25 and predictions[top_class] > predictions[top_k[1]] + 0.1:
                    if self.class_mapping:
                        result = self.class_mapping[self.labels[top_class]]
                    else:
                        result = top_class
                if state is not None and result != state:
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if not os.path.exists('gt'):
                        os.makedirs('gt')
                    cv2.imwrite('gt/%d_Detected_%d_as_%d.png' % (self.count, state, result), bgr)
                    self.count += 1

            return result


def test_file(file_name, classifier):
    bgr = cv2.imread(file_name)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print ('file: %s, detected class: %s' % (file_name, classifier.get_classification(rgb)))


if __name__ == '__main__':

    # Create classifier from persistence
    model = \
        {
            'input_width': 128,
            'input_height': 128,
            'input_depth': 3,
            'resized_input_tensor_name': "input:0",
            'output_tensor_name': "final_result:0",
            'model_file_name': "graph.pb",
            'labels_file_name': "labels.txt",
            'input_mean': 127.5,
            'input_std': 127.5
        }
    mapping = \
        {
            'none': 'TrafficLight.UNKNOWN',
            'green': 'TrafficLight.GREEN',
            'yellow': 'TrafficLight.YELLOW',
            'red': 'TrafficLight.RED'
        }
    classifier = TLClassifier(model, mapping)

    # Test on image
    test_file("test_red.png", classifier)
    test_file("test_yellow.png", classifier)
    test_file("test_green.png", classifier)

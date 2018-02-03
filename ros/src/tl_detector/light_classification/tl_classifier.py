from styx_msgs.msg import TrafficLight
import tensorflow as tf
import scipy.misc
import cv2
import numpy as np
import time
import os


class TLClassifier(object):
    def __init__(self, model_info=None, collect_training_data=False):
        self.model_info = model_info
        self.collect_training_data = collect_training_data

        if self.model_info is not None:
            # Load the persisted model into default graph
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
        tl_color = "Unknown"
        if state:
            if state == TrafficLight.GREEN:
                tl_color = "Green"
            elif state == TrafficLight.YELLOW:
                tl_color = "Yellow"
            elif state == TrafficLight.RED:
                tl_color = "Red"

        if tl_color:
            directory = "gt/{}".format(tl_color)
            if not os.path.exists(directory):
                os.makedirs(directory)

            time_str = time.strftime("%Y%m%d-%H%M%S")
            file_name = directory + "/img_{}.png".format(time_str)
            cv2.imwrite(file_name, image)


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
            # self.save_training_img(image, state)
            return state
        elif self.model_info is not None:
            with tf.Session() as sess:
                # Preprocess
                image = scipy.misc.imresize(image, self.input_size, 'bilinear')
                image = np.squeeze((image.astype('Float32') - self.input_mean) / self.input_std)

                # Classifiy
                result_tensor = sess.graph.get_tensor_by_name(self.model_info['output_tensor_name'])
                predictions, = sess.run(result_tensor, {self.model_info['resized_input_tensor_name']: [image]})
                top_k = predictions.argsort()[-2:][::-1]

                for node_id in top_k:
                    human_string = self.labels[node_id]
                    score = predictions[node_id]
                    print('%s (score = %.5f)' % (human_string, score))

                if predictions[top_k[0]] > 0.5 and predictions[top_k[0]] > predictions[top_k[1]] + 0.15:
                    return top_k[0]

        return TrafficLight.UNKNOWN


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
    classifier = TLClassifier(model)

    # Test on image
    bgr = cv2.imread("test_daisy.jpg")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    classifier.get_classification(rgb)

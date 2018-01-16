from styx_msgs.msg import TrafficLight
import cv2
import time
import os
import rospy


class TLClassifier(object):
    def __init__(self, collect_training_data=False):

        #TODO load classifier
        self.collect_training_data = collect_training_data

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

    def get_classification(self, image, state=None):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light
            state: Current state of traffic light (ground truth only used for training)

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO implement light color prediction
        #rospy.loginfo("Received message")

        if self.collect_training_data:
            self.save_training_img(image, state)
            return state

        return TrafficLight.UNKNOWN

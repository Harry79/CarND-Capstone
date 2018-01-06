#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.np_waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.np_waypoints = np.array([[wp.pose.pose.position.x, wp.pose.pose.position.y,
                                       wp.pose.pose.position.z] for wp in waypoints.waypoints])

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))

        if not light_wp == -1:
            #rospy.loginfo("Currently detected red light at ind {}".format(self.last_wp))
            pass

        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        min_ind = None
        min_dist = 1e+100

        for ind, wp in enumerate(self.waypoints.waypoints):
            d = np.linalg.norm(pose - np.array([wp.pose.pose.position.x, wp.pose.pose.position.y,
                                                wp.pose.pose.position.z]))
            if d < min_dist:
                min_ind = ind
                min_dist = d

        return min_ind

    def get_closest_traffic_light(self, light_position):

        if not self.lights:
            return None

        min_light_dist = 1e+10
        closest_light = None

        ecl = lambda a, b: math.sqrt((a.x - b[0]) ** 2 + (a.y - b[1]) ** 2)
        for light in self.lights:
            light_pose = light.pose.pose
            dist = ecl(light_pose.position, light_position)
            if dist < min_light_dist:
                min_light_dist = dist
                closest_light = light

        return closest_light

    def get_light_state(self, light, ):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_pose = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        if self.pose is not None and self.waypoints:

            # Get waypoint clostest to current vehicle position
            wp_position_ind = self.get_closest_waypoint(self.pose)
            wp_position = self.waypoints.waypoints[wp_position_ind]

            detection_distance = 50
            min_light_dist = 1e+10
            closest_light_index = None
            ecl = lambda a, b: math.sqrt((a.x - b[0]) ** 2 + (a.y - b[1]) ** 2)
            for ind, light_position in enumerate(stop_line_positions):
                light_x = light_position[0]
                car_position = wp_position.pose.pose.position
                dist = abs(car_position.x - light_x)
                if dist < detection_distance:
                    euc_dist = ecl(car_position, light_position)
                    if euc_dist < detection_distance and euc_dist < min_light_dist:
                        min_light_dist = euc_dist
                        closest_light_index = ind

            #rospy.loginfo("Distance to closest traffic light {} is {}".format(closest_light_index, min_light_dist))

            if closest_light_index != None:
                light = self.get_closest_traffic_light(stop_line_positions[closest_light_index])
                light_pose = np.array([stop_line_positions[closest_light_index][0],
                                       stop_line_positions[closest_light_index][1], 0])

        if light:
            state = self.get_light_state(light)
            light_wp_ind = self.get_closest_waypoint(light_pose)
            light_wp = self.waypoints.waypoints[light_wp_ind]

            # Todo: Return state of traffic light received from classification algorithm
            return light_wp_ind, light.state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

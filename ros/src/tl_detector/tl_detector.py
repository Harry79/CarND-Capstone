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

# TODO: Move this to config file
STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    """

    """
    def __init__(self):
        rospy.init_node('tl_detector')
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.stop_lines = self.config['stop_line_positions']
        self.lights = []


        self.has_image = False

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

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        #rospy.spin()

        # Enter processing loop
        self.loop()

    # Main loop
    def loop(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # TODO remove when images start working correctly in sim
            light_wp, state = self.process_traffic_lights()
            if state == TrafficLight.RED or state == TrafficLight.YELLOW:
                self.upcoming_red_light_pub.publish(Int32(light_wp))
                #rospy.loginfo("Currently detected red light at ind {}".format(light_wp))
            rate.sleep()

    def pose_cb(self, msg):
        #print(msg);
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        #TODO remove early return when images start working correctly in sim
        return

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

    @staticmethod
    def get_distance(first, second):
        """
        Get eucledian distance between two point
        :param first: np array for first array
        :param second: np array for second array
        :return: eucledian distance
        """
        return np.linalg.norm(first - second)

    def get_closest_waypoint_xyz(self, pose):
        min_ind = None
        min_dist = 1e+100

        for ind, wp in enumerate(self.waypoints.waypoints):
            d = self.get_distance(pose, np.array([wp.pose.pose.position.x, wp.pose.pose.position.y,
                                                  wp.pose.pose.position.z]))
            if d < min_dist:
                min_ind = ind
                min_dist = d

        return min_ind

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.get_closest_waypoint_xyz(np.array([pose.position.x, pose.position.y, pose.position.z]))

    def get_closest_traffic_light(self, light_position):
        """
        :param light_position:
        :return:
        """
        if not self.lights:
            return None

        min_light_dist = 1e+10
        closest_light = None
        light_position_arr = np.array([light_position[0], light_position[1]])

        for light in self.lights:
            light_pose = np.array([light.pose.pose.position.x, light.pose.pose.position.y])
            dist = self.get_distance(light_pose, light_position_arr) #ecl(light_pose, light_position)
            if dist < min_light_dist:
                min_light_dist = dist
                closest_light = light

        return closest_light

    def get_closest_stop_line(self, position, detection_distance=50):
        # Init variables for search
        min_light_dist = 1e+10
        closest_stop_line = None

        # Convert car position to np array
        pose_arr = np.array([position.x, position.y])

        for stop_line in self.stop_lines:
            stop_line_position = np.array([stop_line[0], stop_line[1]])
            dist = self.get_distance(stop_line_position, pose_arr)
            if dist < detection_distance and dist < min_light_dist:
                min_light_dist = dist
                closest_stop_line = stop_line

        return closest_stop_line


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Check if there is really an image
        if not self.has_image:
            # TODO Remove return of ground truth state, when camera image works
            return light.state
            #return TrafficLight.UNKNOWN

        # Convert image to OpenCv format
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification from DNN
        return self.light_classifier.get_classification(cv_image, light.state)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        light = None

        # Check if pose is valid and waypoints are set
        if self.pose is not None and self.waypoints:

            print(self.lights)

            # List of positions that correspond to the line to stop in front of for a given intersection
            stop_line_positions = self.config['stop_line_positions']
            # Get waypoint closest to current vehicle position
            wp_position_ind = self.get_closest_waypoint(self.pose.pose)

            #TODO find the closest visible traffic light (if one exists)

            
            wp_position = self.waypoints.waypoints[wp_position_ind].pose.pose.position
            # Get stop line closest to waypoint position
            stop_line = self.get_closest_stop_line(wp_position)
            
            
            # TODO Cant use delta x > 0 for global world coordinates,
            # TODO need to find other way to ensure that traffic light is ahead
            # TODO Also moved code into function "get_closest_stop_line"
            """
            # Init variables for search
            detection_distance = 50
            min_light_dist = 1e+10
            closest_light_index = None

            # Convert car position to np array
            car_position = np.array([wp_position.pose.pose.position.x, wp_position.pose.pose.position.y])

            for ind, light_position in enumerate(stop_line_positions):
                light_x = light_position[0]
                light_position_arr = np.array([light_position[0], light_position[1]])

                # Use signed distance to detect traffic lights in front of the vehicle
                dist = light_x - car_position[0]
                if dist > 0:
                    euc_dist = self.get_distance(car_position, light_position_arr)
                    if euc_dist < detection_distance and euc_dist < min_light_dist:
                        min_light_dist = euc_dist
                        closest_light_index = ind
            #rospy.loginfo("Distance to closest traffic light {} is {}".format(closest_light_index, min_light_dist))
            """

            
            if stop_line is not None:
                # Get traffic light closest to stop line position
                light = self.get_closest_traffic_light(stop_line)
                if light is not None:
                    state = self.get_light_state(light)
                    stop_line_position = np.array([stop_line[0], stop_line[1], 0])
                    light_wp_ind = self.get_closest_waypoint_xyz(stop_line_position)
                    return light_wp_ind, state
            


#        if light:
#            state = self.get_light_state(light)
#            return light_wp, state
#        self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

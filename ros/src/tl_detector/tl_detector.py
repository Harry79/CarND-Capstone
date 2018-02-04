#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import os
import tf
import cv2
import yaml
import math
import numpy as np

# TODO: Move this to config file
STATE_COUNT_THRESHOLD = 3

'''
/vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
helps you acquire an accurate ground truth data source for the traffic light
classifier by sending the current color state of all traffic lights in the
simulator. When testing on the vehicle, the color state will not be available. You'll need to
rely on the position of the light and the camera image to predict it.
'''
class TLDetector(object):

    def __init__(self):
        rospy.init_node('tl_detector')
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # Setup buffers
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.stop_lines = self.config['stop_line_positions']
        self.lights = []

        # Setup subscribers/publishers
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.bridge = CvBridge()

        # Setup classifier
        model = \
            {
                'input_width': 128,
                'input_height': 128,
                'input_depth': 3,
                'resized_input_tensor_name': "input:0",
                'output_tensor_name': "final_result:0",
                'model_file_name': "light_classification/graph.pb",
                'labels_file_name': "light_classification/labels.txt",
                'input_mean': 127.5,
                'input_std': 127.5
            }
        mapping = \
            {
                'none': TrafficLight.UNKNOWN,
                'green': TrafficLight.GREEN,
                'yellow': TrafficLight.YELLOW,
                'red': TrafficLight.RED
            }
        self.light_classifier = TLClassifier(model, mapping, True)
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Setup camera projection
        self.img_count = 0
        self.img_dump_dir = ''  # Set to <out_dir_name> to enabled frame dumping
        self.img_size = (self.config['camera_info']['image_width'], self.config['camera_info']['image_height'])
        self.focal_length = (self.config['camera_info']['focal_length_x'], self.config['camera_info']['focal_length_y'])
        self.has_image = False
        self.listener = tf.TransformListener()

        # Setup simulator workaround
        self.workaround_sim = False
        if self.workaround_sim:
            self.loop()
        else:
            rospy.spin()

    # Workaround loop
    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # Signal eventual traffic lights
            self.has_image = True
            self.camera_image = None
            light_wp, state = self.process_traffic_lights()

            if state == TrafficLight.RED or state == TrafficLight.YELLOW:
                self.upcoming_red_light_pub.publish(Int32(light_wp))
                #rospy.loginfo("Currently detected red light at ind {}".format(light_wp))
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    """
    Identifies red lights in the incoming camera image and publishes the index
    of the waypoint closest to the red light's stop line to /traffic_waypoint
    Args:
    msg (Image): image from car-mounted camera
    """
    def image_cb(self, msg):
        if not self.workaround_sim:
            # Signal eventual traffic lights
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

    """
    Get eucledian distance between two point
    :param first: np array for first array
    :param second: np array for second array
    :return: eucledian distance
    """
    @staticmethod
    def get_distance(first, second):
        return np.linalg.norm(first - second)

    def dump_frame(self, frame_image, x1, y1, x2, y2, color):
        if len(self.img_dump_dir) > 0:
            if not os.path.exists(self.img_dump_dir):
                os.makedirs(self.img_dump_dir)
            bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color)
            self.img_count = self.img_count + 1
            filename = self.img_dump_dir + "/cvimg-%02i.png" % self.img_count
            cv2.imwrite(filename, bgr)

    """
    Identifies the closest path waypoint to the given position
    https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
    Args:
    pose (Pose): position to match a waypoint to
    Returns:
    int: index of the closest waypoint in self.waypoints
    """
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
        return self.get_closest_waypoint_xyz(np.array([pose.position.x, pose.position.y, pose.position.z]))

    """
    :param light_position:
    :return: List of traffic lights in vicinity, sorted by distance
    """
    def get_closest_traffic_lights(self, pose, detection_distance=100):
        if not self.lights:
            return None
        position_arr = np.array([pose.position.x, pose.position.y])
        lights = []
        for light in self.lights:
            light_pose = np.array([light.pose.pose.position.x, light.pose.pose.position.y])
            dist = self.get_distance(light_pose, position_arr)
            if dist < detection_distance:
                lights.append(light)
        lights.sort(key=lambda l: self.get_distance(np.array([l.pose.pose.position.x, l.pose.pose.position.y]), position_arr))
        return lights

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

    """Determines the current color of the traffic light
    Args:
    light (TrafficLight): light to classify
    Returns:
    int: ID of traffic light color (specified in styx_msgs/TrafficLight)
    """
    def get_light_state(self, light_projection):
        # Check if there is really an image
        if not self.has_image:
            return light_projection[0].state if self.workaround_sim else TrafficLight.UNKNOWN

        # Convert image to OpenCv format
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # Estimate bounds
        proj = light_projection[1]
        if proj[2] != 0:
            x1 = max(0, min(int(proj[0] - 3.0 * 320.0 * proj[2]), self.img_size[0] - 1))
            y1 = max(0, min(int(proj[1] - 1000.0      * proj[2]), self.img_size[1] - 1))
            x2 = max(0, min(int(proj[0] + 3.0 * 320.0 * proj[2]), self.img_size[0] - 1))
            y2 = max(0, min(int(proj[1] + 5000.0      * proj[2]), self.img_size[1] - 1))
            self.dump_frame(cv_image, x1, y1, x2, y2, (255, 0, 0))

            # Get classification from DNN
            if abs(x2 - x1) > 32 and abs(y2 - y1) > 32:
                return self.light_classifier.get_classification(cv_image[y1:y2, x1:x2], light_projection[0].state)
        return TrafficLight.UNKNOWN

    def project_traffic_light_to_view(self, lights):
        # print("HS: listener")
        # print(self.listener);
        # print("HS: pose")
        # print(self.pose);
        # tf::StampedTransform transform;
        t = self.listener.getLatestCommonTime("/world", "/base_link")
        (trans, rot) = self.listener.lookupTransform("/world", "/base_link", t)

        image_width = self.img_size[0]
        image_height = self.img_size[1]
        trans_mat = tf.transformations.translation_matrix(trans)
        rot_mat = tf.transformations.quaternion_matrix(rot)
        n = 1.0  # you may want to adjust this if you want to look closer
        f = 100.0  # you may want to adjust this if you want to took further
        t = image_height / self.focal_length[1]
        b = -t
        r = image_width / self.focal_length[0]
        l = -r

        # http://www.glprogramming.com/red/appendixf.html
        # proj_mat = np.matrix([[ 2*n/(r-l), 0,         (r+l)/(r-l), 0           ],
        #                      [ 0,         2*n/(t-b), (t+b)/(t-b), 0           ],
        #                      [ 0,         0,        -(f+n)/(f-n), -2*f*n/(f-n)],
        #                      [ 0,         0,        -1,           0           ]]);
        proj_mat = np.array([[-(r + l) / (r - l), -2.0 * n / (r - l), 0.0, 0.0],
                             [-(t + b) / (t - b), 0.0, 2.0 * n / (t - b), 0.0],
                             [(f + n) / (f - n), 0.0, 0.0, -2.0 * f * n / (f - n)],
                             [1.0, 0.0, 0.0, 0.0]])

        # the camera is obviously looking upwards
        al = 7.0 * math.pi / 180.0  # adjust the angle
        cosa = math.cos(al)
        sina = math.sin(al)
        bl = 0.7 * math.pi / 180.0  # adjust the angle
        cosb = math.cos(bl)
        sinb = math.sin(bl)
        rotlookat_mat = np.array([[cosa, 0.0, sina, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [-sina, 0.0, cosa, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        rotlookatz_mat = np.array([[cosb, -sinb, 0.0, 0.0],
                                   [sinb, cosb, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]])
        mat = rotlookatz_mat.dot(rotlookat_mat.dot(np.linalg.inv(np.dot(trans_mat, rot_mat))))
        # mat = np.linalg.inv(np.dot(trans_mat, rot_mat))
        # mat = proj_mat.dot(np.linalg.inv(np.dot(trans_mat, rot_mat)))

        for light in lights:
            transformed = mat.dot(
                np.array([light.pose.pose.position.x, light.pose.pose.position.y, light.pose.pose.position.z, 1]))
            projected = proj_mat.dot(transformed)
            # print(proj_mat)
            #                tansformed = self.pose.pose * light.pose.pose.position;
            # print(mat);
            # print(transformed)
            # print(projected/projected[3])

            projected = projected / projected[3]
            # clip
            if (-1 < projected[0] and 1 > projected[0] and
                        -1 < projected[1] and 1 > projected[1] and
                        -1 < projected[2] and 1 > projected[2]):
                projx = projected[0] * image_width / 2.0 + image_width / 2.0
                projy = -projected[1] * image_height / 2.0 + image_height / 2.0
                projs = 1.0 / transformed[0]
                # print("screenpos = %g %g scale = %g",
                #      self.projx,
                #      self.projy,
                #      self.projs)
                # else:
                # print("out")
                # light_pose = tf.Vector3Stamped()
                # light_pose.vector.x = light.pose.pose.position.x
                # light_pose.vector.y = light.pose.pose.position.y
                # light_pose.vector.z = light.pose.pose.position.z
                return light, (projx, projy, projs)
        return None, (0, 0, 0)


    """
    Finds closest visible traffic light, if one exists, and determines its
    location and color
    Returns:
    int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
    int: ID of traffic light color (specified in styx_msgs/TrafficLight)
    """
    def process_traffic_lights(self):
        self.projs = 0
        
        # Check if pose is valid and waypoints are set
        if self.pose and self.waypoints and self.lights and self.stop_lines:

            # Get lights around vicinity
            lights = self.get_closest_traffic_lights(self.pose.pose)

            if len(lights) > 0:
                # Get closest projected light in view
                light_projection = self.project_traffic_light_to_view(lights)

                if light_projection[0]:
                    # Get state of light
                    state = self.get_light_state(light_projection)

                    # Get stop line closest to light
                    stop_line = self.get_closest_stop_line(light_projection[0].pose.pose.position)

                    if stop_line:
                        stop_line_position = np.array([stop_line[0], stop_line[1], 0])

                        # Get waypoint closest to stopline
                        light_wp_ind = self.get_closest_waypoint_xyz(stop_line_position)
                        return light_wp_ind, state
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

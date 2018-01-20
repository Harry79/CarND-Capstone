#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


# Tools
def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Add other member variables you need below
        self.base_waypoints = []
        self.final_waypoints = []
        self.cur_pos = PoseStamped()
        self.cur_wp_idx = -1           # Index of wp we want to move to in the current loop
        self.last_wp_idx = -1          # Index of wp from the previous loop
        self.next_light_idx = -1    # Index of the next light (-1 if no upcoming light)
        self.max_velocity = self.kmph2mps(rospy.get_param('/waypoint_loader/velocity'))

        # Enter processing loop
        self.loop()

    def pose_cb(self, pose):
        self.cur_pos = pose

    def waypoints_cb(self, lane):
        # Simply assign waypoints from lane
        self.base_waypoints = lane.waypoints

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message
        self.next_light_idx = msg.data
        rospy.loginfo('callback light idx received: ' + str(self.next_light_idx))

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    # Returns beeline distance along waypoints in between wp1 and wp2
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += distance(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

    # Main loop
    def loop(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.cur_pos.header.seq > 0 and len(self.base_waypoints) > 0:
		max_index = len(self.base_waypoints)

                if len(self.final_waypoints) > 0:
                    # Final waypoints already exist so use as candidates for the next round
		    waypoint_candidates = self.final_waypoints
                else:
                    # No Final waypoints so use the base waypoints
                    waypoint_candidates = self.base_waypoints

                # Find nearest base waypoint (ignore heading)
                nearest_wp = min(
                    waypoint_candidates,
                    key=lambda wp, pos=self.cur_pos: distance(pos.pose.position, wp.pose.pose.position))
                # Make the nearest waypoint to be the current one and get its index
                self.cur_wp_idx = self.base_waypoints.index(nearest_wp)
		rospy.loginfo('current waypoint velocity: ' + str(self.get_waypoint_velocity(nearest_wp)))

                # Generate & publish new final waypoints, if we moved
                if self.cur_wp_idx != self.last_wp_idx:
                    del self.final_waypoints[:]
                    for i in range(0, LOOKAHEAD_WPS):
                        self.final_waypoints.append(self.base_waypoints[(self.cur_wp_idx + i) % max_index])
                    rospy.loginfo('next_light_idx: ' + str(self.next_light_idx))
                    if self.next_light_idx <> -1: #if there is an upcoming red light, set wp velocities to 0
                        rospy.loginfo('next_light_idx <> -1')
                        for i in range(0, LOOKAHEAD_WPS):
                            self.set_waypoint_velocity(self.final_waypoints, i, 0)
                    else:       #if there is no upcoming red light, set wp velocities to the max in rosparam
                        rospy.loginfo('next_light_idx = -1')
                        for i in range(0, len(self.final_waypoints)):
                            self.set_waypoint_velocity(self.final_waypoints, i, self.max_velocity)
                    rospy.loginfo('updated final waypoints')
                    self.publish()
                    self.last_wp_idx = self.cur_wp_idx
            rate.sleep()

    # Publish final waypoints
    def publish(self):
        lane = Lane()
        lane.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

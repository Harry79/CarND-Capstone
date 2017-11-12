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
        self.cur_pos = PoseStamped()
        self.cur_wp = -1
        self.last_wp = -1

        # Enter processing loop
        self.loop()

    def pose_cb(self, pose):
        self.cur_pos = pose

    def waypoints_cb(self, lane):
        # Simply assign waypoints from lane
        self.base_waypoints = lane.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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

    # Main loop
    def loop(self):
        rate = rospy.Rate(2)

        while not rospy.is_shutdown():
            if self.cur_pos.header.seq > 0 and len(self.base_waypoints) > 0:
                # Find nearest base waypoint
                nearest_wp = min(
                    self.base_waypoints,
                    key=lambda wp, pos=self.cur_pos: distance(pos.pose.position, wp.pose.pose.position))

                # TODO: Instead of always skipping 1, skip dependent on orientation
                self.cur_wp = self.base_waypoints.index(nearest_wp) + 1

                # Publish new final waypoints, if changed
                if self.cur_wp != self.last_wp:
                    self.publish()
                    self.last_wp = self.cur_wp
            rate.sleep()

    # Publish final waypoints
    def publish(self):
        lane = Lane()
        max_index = len(self.base_waypoints)
        for i in range(0, LOOKAHEAD_WPS):
            lane.waypoints.append(self.base_waypoints[(self.cur_wp + i) % max_index])
        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

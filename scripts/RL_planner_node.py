#!/usr/bin/env python3
from nav_msgs.msg import Odometry
import tf
import sys
import copy
import time
import os
import csv
import math

from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from RL_planner_net import *
import rospy
import roslib
roslib.load_manifest('RL_planner')

class RL_planner_node:
    def __init__(self):
        rospy.init_node('RL_planner_node', anonymous=True)
        self.target_pos_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.target_callback)
        self.ps_map = PoseStamped()
        self.ps_map.header.frame_id = 'map'
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.scan = LaserScan()

    def target_callback(self, msg):
        self.ps_map.pose = msg.pose

    def scan_callback(self, msg):
        self.scan.ranges = msg.ranges
    
    def tf_target2robot(self):
        try:
            (trans, rot) = listener.lookupTransform('/odom', '/base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        ps_base = listener.transformPose('/odom', ps_map)

        dx = ps_base.pose.position.x - trans[0]
        dy = ps_base.pose.position.y - trans[1]
        dist = math.sqrt(dx*dx + dy*dy)

        angle = math.atan2(dy, dx)
        quat = tf.transformations.quaternion_from_euler(0, 0, angle)
        (_, _, yaw) = tf.transformations.euler_from_quaternion(quat)
        angle_diff = yaw - tf.transformations.euler_from_quaternion(rot)[2]

        print('Distance: %.2f m' % dist)
        print('Angle: %.2f rad' % angle_diff)

        return dist, angle_diff
    
    def loop(self):
        pass

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
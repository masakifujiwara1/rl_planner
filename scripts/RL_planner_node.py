#!/usr/bin/env python3
import tf
import sys
import copy
import time
import os
import csv
import math
import random

from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Twist, Pose
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry

# from RL_planner_net import *
import rospy
import roslib
roslib.load_manifest('rl_planner')

# temp param
GAZEBO_X = -2.0
GAZEBO_Y = -0.5

MODEL_NAME = 'turtlebot3_burger'


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

class Gazebo_env:
    def __init__(self):
        # rospy.wait_for_service('/gazebo/rest_world')
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_pos = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def reset_env(self):
        self.reset_world()

    def get_state(self):
        resp = self.get_pos(MODEL_NAME, '')
        get_state = Pose()
        get_state = resp.pose
        return get_state

    def set_ang(self):
        model_state = ModelState()
        random_ang = random.uniform(-3.14, 3.14)
        # print(random_ang)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, random_ang)
        state = self.get_state()
        state.orientation.x = quaternion[0]   
        state.orientation.y = quaternion[1]
        state.orientation.z = quaternion[2]
        state.orientation.w = quaternion[3]
        model_state.model_name = MODEL_NAME
        model_state.pose = state
        self.set_pos(model_state)

    def reset_and_set(self):
        self.reset_env()
        self.set_ang()

if __name__ == '__main__':
    # rg = nav_cloning_node()
    # DURATION = 0.2
    # r = rospy.Rate(1 / DURATION)
    # while not rospy.is_shutdown():
    #     rg.loop()
    #     r.sleep()
    rs = RL_planner_node()
    rg = Gazebo_env()
    rg.reset_and_set()
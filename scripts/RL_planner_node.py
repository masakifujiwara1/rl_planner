#!/usr/bin/env python3
import tf
import sys
import copy
import time
import os
import csv
import math
import random
import numpy as np

from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Twist, Pose
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry

from RL_planner_net_without_action_scale import *
import rospy
import roslib
roslib.load_manifest('rl_planner')

MODEL_NAME = 'turtlebot3_burger'

# to do random target position 
X = 1.5
Y = 1.5

class RL_planner_node:
    def __init__(self):
        rospy.init_node('RL_planner_node', anonymous=True)
        self.target_pos_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.target_callback)
        self.target_pos_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.ps_map = PoseStamped()
        self.ps_map.header.frame_id = 'map'
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.scan = LaserScan()
        self.listener = tf.TransformListener()
        self.wait_scan = False
        self.min_s = 0

        self.gazebo_env = Gazebo_env()

        self.end_steps_flag = False
        self.eval_flag = False

        self.i_episode = 1
        self.eval_count = 0
        self.args = {
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'seed': 123456,
            'batch_size': 256,
            'hiden_dim': 256,
            'start_steps': 1,
            'target_update_interval': 1,
            'memory_size': 100000,
            'epochs': 100,
            'eval_interval': 1000,
            'end_step': 500,
        }
        self.reward_args = {
            'r_arrive': 10.0,
            'r_collision': -10.0,
            'Cr': 10.0,
            'Cd': 1, # 直線で何m以内で到達したとみなすか
            'Co': 0.15, # 何m以内に障害物があれば衝突したとみなすか
            'r_position': -5.0,
            'ramda_p': 1,
            'ramda_w': 1
        }
        self.train = Trains(self.args, self.reward_args)
        self.make_state = MakeState()

        self.gazebo_env.reset_env()

        self.hand_set_target(X, Y)
        # print(self.ps_map)
        time.sleep(1)

    def target_callback(self, msg):
        self.ps_map.pose = msg.pose

    def scan_callback(self, msg):
        self.scan.ranges = msg.ranges
        scan_ranges = np.array(self.scan.ranges)
        scan_ranges[np.isinf(scan_ranges)] = msg.range_max
        # print(len(scan_ranges))
        self.scan.ranges = scan_ranges
        self.min_s = min(self.scan.ranges)
        self.wait_scan = True

    def hand_set_target(self, x, y):
        self.ps_map.pose.position.x = x
        self.ps_map.pose.position.y = y
    
    def tf_target2robot(self):
        try:
            (trans, rot) = self.listener.lookupTransform('/odom', '/base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        ps_base = self.listener.transformPose('/odom', self.ps_map)

        dx = ps_base.pose.position.x - trans[0]
        dy = ps_base.pose.position.y - trans[1]
        dist = math.sqrt(dx*dx + dy*dy)

        angle = math.atan2(dy, dx)
        quat = tf.transformations.quaternion_from_euler(0, 0, angle)
        (_, _, yaw) = tf.transformations.euler_from_quaternion(quat)
        angle_diff = yaw - tf.transformations.euler_from_quaternion(rot)[2]

        # print('Distance: %.2f m' % dist)
        # print('Angle: %.2f rad' % angle_diff)

        target_l = [dist, angle_diff]
        target_l = np.array(target_l)

        return target_l

    def next_episode(self):
        # twist reset
        self.train.action_twist.linear.x = 0
        self.train.action_twist.angular.z = 0
        self.train.action_pub.publish(self.train.action_twist)
        # target set
        self.hand_set_target(X, Y)
        # gazebo reset
        self.gazebo_env.reset_and_set()
        time.sleep(0.5)
    
    def loop(self):
        if not self.wait_scan:
            return
        self.target_pos_pub.publish(self.ps_map)
        target_l = self.tf_target2robot()
        state = self.make_state.toState(np.array(list(self.scan.ranges)), target_l)

        # #check steps
        # self.end_steps_flag = self.train.check_steps(target_l, self.min_s)

        if self.i_episode == self.args['epochs'] + 1:
            # loop end
            pass

        if not self.eval_flag:
            if self.end_steps_flag:

                self.train.episode_reward_list.append(self.train.episode_reward)
                if self.i_episode % self.args['eval_interval'] == 0:
                    self.eval_flag = True
                    self.eval_count = 0
                
                # env reset
                # self.gazebo_env.reset_and_set()
                self.next_episode()
                self.train.n_steps = 0
                self.train.episode_reward = 0
                self.train.reward = 0
                self.i_episode += 1
                self.train.init = True

                self.end_steps_flag = False
            else:
                self.train.while_func(state, target_l, self.min_s)
                print("Train mode Episode: {}, Step: {}, Step reward: {:.2f}".format(self.i_episode, self.train.n_steps, self.train.reward))
        else:
            if self.eval_count == self.args['eval_interval']:
                self.eval_flag = False
                print("Episode: {}, Eval Avg. Reward: {:.0f}".format(self.i_episode, self.train.avg_reward))
            else:
                if self.end_steps_flag:
                    self.train.avg_reward /= self.args['eval_interval']
                    self.train.eval_reward_list.append(self.train.avg_reward)

                    # print("Episode: {}, Eval Avg. Reward: {:.0f}".format(self.i_episode, self.train.avg_reward))

                    # env reset
                    # self.gazebo_env.reset_and_set()
                    self.next_episode()
                    self.eval_count += 1
                    self.train.n_steps = 0
                    self.train.episode_reward = 0
                    self.train.reward = 0
                    self.train.init = True

                    self.end_steps_flag = False
                else:
                    # eval func
                    self.train.evaluate_func(state, target_l, self.min_s)
                    print("Evaluate mode Episode: {}, Step: {}, Step reward: {:.0f}".format(self.i_episode, self.train.n_steps, self.train.episode_reward))

        # print("Episode: {}, Step: {}, Step reward: {:.0f}".format(self.i_episode, self.train.n_steps, self.train.episode_reward))
        #check steps
        self.end_steps_flag = self.train.check_steps(target_l, self.min_s)

class Trains:
    def __init__(self, args, reward_args):
        self.args = args
        self.reward_args = reward_args
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        # print(device)
        self.agent = ActorCriticModel(args=args, device=device)
        self.memory = ReplayMemory(args['memory_size'])
        self.action_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.action_twist = Twist()

        self.episode_reward_list = []
        self.eval_reward_list = []

        self.n_steps = 0
        self.n_update = 0

        self.episode_reward = 0
        self.done = False
        # self.init = True
        self.action = [0.0, 0.0]
        self.init = True
        self.avg_reward = 0.
        self.reward = 0
        # self.state = 1 # robotに対する目標位置

        self.old_action = [0., 0.]
        self.old_state = 0
        self.old_target_l = 0
    
    def while_func(self, state, target_l, min_s):
        if self.args['start_steps'] > self.n_steps:
            # action = env.action_space.sample()
            self.action[0] = random.uniform(0.1, 0.5)
            self.action[1] = random.uniform(-1.0, 1.0)
        else:
            self.action = self.agent.select_action(self.old_state)

        # print(self.action)

        if len(self.memory) > self.args['batch_size']:
            self.agent.update_parameters(self.memory, self.args['batch_size'], self.n_update)
            self.n_update += 1

        # env update
        self.action_twist.linear.x = abs(self.action[0])
        self.action_twist.angular.z = self.action[1]
        self.action_pub.publish(self.action_twist)

        if not self.init:
            # next_state, reward, done, _ = env.step(action)

            self.reward = self.calc_reward(target_l, min_s, self.old_target_l)

            self.n_steps += 1
            self.episode_reward += self.reward

            self.memory.push(state=self.old_state, action=self.old_action, reward=self.reward, next_state=state, mask=float(not self.done))

        # state = next_state
        self.old_state = state
        self.old_action = self.action
        self.old_target_l = target_l

        self.init = False

    def evaluate_func(self, state, target_l, min_s):
        with torch.no_grad():
            self.action = self.agent.select_action(state, evaluate=True)

        self.action_twist.linear.x = self.action[0]
        self.action_twist.angular.z = self.action[1]
        self.action_pub.publish(self.action_twist)

        if not self.init:
            self.reawrd = self.calc_reward(target_l, min_s, self.old_target_l)
            self.n_steps += 1
            self.episode_reward += self.reawrd

    def calc_reward(self, target_l, min_s, old_target_l):
        # rt
        if target_l[0] < self.reward_args['Cd']:
            rt = self.reward_args['r_arrive']
        elif min_s < self.reward_args['Co'] + 0.01:
            rt = self.reward_args['r_collision']
        else:
            # rt = self.reward_args['Cr'] * (old_target_l[0] - target_l[0])
            rt = self.reward_args['Cr'] * (old_target_l[0] - target_l[0])
        
        # rpt
        if (old_target_l[0] - target_l[0]) == 0:
            rpt = self.reward_args['r_position']
        else:
            # 0.1m/sで近づかないとペナルティ
            # rpt =  - self.reward_args['Cr'] * (0.1 / 5)
            # rpt =  - (0.1 / 5)
            rpt = 0

        # rwt
        rwt = -1 * abs(target_l[1])

        self.reward = rt + self.reward_args['ramda_p'] * rpt + self.reward_args['ramda_w'] * rwt

        return self.reward

    def check_steps(self, target_l, min_s):
        flag = False
        if self.args['end_step'] == self.n_steps:
            flag = True
        elif target_l[0] < self.reward_args['Cd']:
            flag = True
        elif min_s < self.reward_args['Co']:
            flag = True
        return flag

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
    rg = RL_planner_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()

    # rs = RL_planner_node()
    # rg = Gazebo_env()
    # rg.reset_and_set()
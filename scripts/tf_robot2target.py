#!/usr/bin/env python3
import rospy
import tf
from geometry_msgs.msg import PoseStamped
import math

def callback(msg):
    # ロボットの現在位置を取得する
    try:
        (trans, rot) = listener.lookupTransform('/odom', '/base_footprint', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        return

    # 目標位置をbase_footprint座標系に変換する
    ps_map = PoseStamped()
    ps_map.header.frame_id = 'map'
    ps_map.pose = msg.pose
    ps_base = listener.transformPose('/odom', ps_map)

    # 目標位置とロボットの現在位置の差を計算する
    dx = ps_base.pose.position.x - trans[0]
    dy = ps_base.pose.position.y - trans[1]
    dist = math.sqrt(dx*dx + dy*dy)

    # 角度の差を計算する
    angle = math.atan2(dy, dx)
    robot_ang = tf.transformations.euler_from_quaternion(rot)[2]

    if angle < 0:
        angle += math.pi * 2

    if robot_ang < 0:
        robot_ang += math.pi * 2

    angle_diff = angle - robot_ang

    if angle_diff < 0:
        angle_diff += math.pi * 2

    # 結果を表示する
    print('Distance: %.2f m' % dist)
    print('Angle: %.2f rad' % angle_diff)
    # print(angle, robot_ang)

if __name__ == '__main__':
    rospy.init_node('goal_converter')
    listener = tf.TransformListener()
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback)
    rospy.spin()

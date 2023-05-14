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
    quat = tf.transformations.quaternion_from_euler(0, 0, angle)
    (_, _, yaw) = tf.transformations.euler_from_quaternion(quat)
    angle_diff = yaw - tf.transformations.euler_from_quaternion(rot)[2]

    # 結果を表示する
    # print('robot_x: %.2f m' % trans[0])
    # print('robot_y: %.2f m' % trans[1])
    print('Distance: %.2f m' % dist)
    print('Angle: %.2f rad' % angle_diff)

if __name__ == '__main__':
    rospy.init_node('goal_converter')
    listener = tf.TransformListener()
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback)
    rospy.spin()

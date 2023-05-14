#!/usr/bin/env python3
import tf
import rospy

x, y, z = 0.0, 0.0, 0.0
roll, pitch, yaw = 0, 0, 0

rospy.init_node('map2odom')

# map座標系からodom座標系への変換量
translation = (x, y, z) 
rotation = tf.transformations.quaternion_from_euler(roll, pitch, yaw) 

# TransformBroadcasterインスタンスを作成する
br = tf.TransformBroadcaster()

# 一定周期でmapからodomへの変換を放送する
while not rospy.is_shutdown():
    br.sendTransform(translation, rotation, rospy.Time.now(), 'odom', 'map')
    rospy.sleep(0.1)
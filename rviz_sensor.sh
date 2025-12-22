#!/bin/bash

# 2. ouster_ros 노드 실행 (IP는 LiDAR 기본 주소인 192.168.1.1로 설정)
roslaunch ouster_ros sensor.launch sensor_hostname:=192.168.0.49

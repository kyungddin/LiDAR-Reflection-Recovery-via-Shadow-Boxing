#!/bin/bash

# 1. dnsmasq 실행해서 LiDAR에 IP 자동 할당
sudo dnsmasq -C/dev/null -kd -F 192.168.0.0,192.168.0.100 -i enx588694fda289 --bind-dynamic

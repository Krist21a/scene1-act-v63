#!/usr/bin/env bash
set -euo pipefail
cd /root/scene1_act_optimization
set +u  # ROS setup scripts reference optional unset variables.
source /opt/ros/noetic/setup.bash
set -u
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
export PYTHONPATH="/root/LeTools-Learning:${PYTHONPATH:-}"
exec /root/miniforge3/envs/kls_dev/bin/python run_scene1.py "$@"

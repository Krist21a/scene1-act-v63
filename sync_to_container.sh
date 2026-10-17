#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
container_name=kuavo-ros-noetic-s12-ready
sudo -n docker exec "$container_name" mkdir -p /root/scene1_act_optimization
# Explicit file list: never copy output images, ROS bags, checkpoints or caches.
for task_file in scene1 configs config.yaml run_scene1.py capture_scene1.py inspect_rgbd.py look_at_scene1.py calibrate_servo.py validate_tracking.py run.sh README.md tests; do
    sudo -n docker cp "$task_dir/$task_file" "$container_name:/root/scene1_act_optimization/"
done
sudo -n docker exec "$container_name" mkdir -p /root/scene1_act_optimization/diagnostics
for task_file in probe_act.py analyze_handover.py analyze_visual_cycle.py replay_trace.py; do
    sudo -n docker cp "$task_dir/diagnostics/$task_file" "$container_name:/root/scene1_act_optimization/diagnostics/$task_file"
done
sudo -n docker exec "$container_name" bash /root/scene1_act_optimization/run.sh --preflight

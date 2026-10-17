#!/usr/bin/env bash
# Read-only capture of the public simulator window; no ROS or control access.
# Set the viewer to 1792x1484 and inspect the framing before starting a run.
# Send SIGINT to this process to close the recording, then remux MKV to MP4.
set -euo pipefail
output=${1:?Usage: record_third_view.sh output.mkv}
export DISPLAY=${DISPLAY:-:0}
mapfile -t windows < <(xdotool search --name '^MuJoCo : Challenge Cup Scene 1$')
if [[ ${#windows[@]} -ne 1 ]]; then
    echo 'Expected exactly one official Scene 1 viewer window.' >&2
    exit 1
fi
window=${windows[0]}
geometry=$(xwininfo -id "$window")
width=$(awk '/Width:/{print $2}' <<< "$geometry")
height=$(awk '/Height:/{print $2}' <<< "$geometry")
if [[ $width != 1792 || $height != 1484 ]]; then
    echo "Viewer is ${width}x${height}; verify framing at 1792x1484 first." >&2
    exit 1
fi
if [[ -e $output ]]; then
    echo 'Output already exists; refusing to overwrite a recording.' >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$output")"
echo "Recording window $window at $(date -Is); original wall-clock speed."
exec ffmpeg -hide_banner -nostdin -f x11grab -draw_mouse 0 -framerate 30 \
    -window_id "$window" -i "$DISPLAY" -t 5400 \
    -vf 'crop=1248:702:270:300,scale=1280:720' -filter_threads 1 \
    -c:v libx264 -preset veryfast -crf 21 -threads 2 -pix_fmt yuv420p \
    -vsync cfr -r 30 -n "$output"

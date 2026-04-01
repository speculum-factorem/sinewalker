# ROS2 Bridge (WebSocket <-> ROS2)

This bridge receives motor commands from the frontend `hardware-websocket` backend and publishes them to ROS2 topics.
It also subscribes to ROS2 sensor topics and streams sensor state back to WebSocket clients.
For real hardware, use the safety controller node below between bridge and drivers.

## What it does

- WebSocket server (default: `ws://0.0.0.0:8765`)
- Receives payload type:
  - `robot-io-tick` (from frontend)
- Publishes to ROS2:
  - `/robot_io/joint_target_position` (`std_msgs/Float64MultiArray`)
  - `/robot_io/joint_target_velocity` (`std_msgs/Float64MultiArray`)
  - `/robot_io/joint_target_effort` (`std_msgs/Float64MultiArray`)
  - `/robot_io/joint_command_json` (`std_msgs/String`)
- Subscribes from ROS2:
  - `/imu/data` (`sensor_msgs/Imu`)
  - `/joint_states` (`sensor_msgs/JointState`)
  - `/robot/velocity` (`geometry_msgs/Twist`)
- Sends sensor updates to WebSocket clients:
  - `sensor-state`

## Expected frontend message

The frontend sends:

```json
{
  "type": "robot-io-tick",
  "ts": 1710000000000,
  "timeSeconds": 1.23,
  "state": {
    "simTimeSeconds": 1.23,
    "com": {"x": 0, "y": 0, "z": 0},
    "linearVelocity": {"x": 0, "y": 0, "z": 0}
  },
  "commands": [
    {
      "jointId": "leg-0",
      "index": 0,
      "targetPosition": 0.12,
      "targetVelocity": 1.4,
      "targetEffort": 3.2
    }
  ]
}
```

## Run

1. Source ROS2:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
```

2. Install Python deps (if needed):

```bash
pip install -r robotics/ros2_bridge/requirements.txt
```

3. Start bridge:

```bash
python robotics/ros2_bridge/bridge_node.py --ws-host 0.0.0.0 --ws-port 8765
```

4. In frontend, select:
   - `Robot I/O backend` -> `hardware backend (WebSocket)`
   - `Hardware endpoint` -> `ws://localhost:8765`

## Safety Controller (watchdog + e-stop + clamping + joint mapping)

Added node:

- `robotics/ros2_bridge/safety_controller_node.py`

It subscribes to:

- `/robot_io/joint_command_json` (`std_msgs/String`) from the bridge
- `/robot_io/e_stop` (`std_msgs/Bool`) external emergency stop

It publishes safe outputs:

- `/robot_io/safe_joint_target_position` (`std_msgs/Float64MultiArray`)
- `/robot_io/safe_joint_target_velocity` (`std_msgs/Float64MultiArray`)
- `/robot_io/safe_joint_target_effort` (`std_msgs/Float64MultiArray`)
- `/robot_io/safe_joint_targets` (`sensor_msgs/JointState`)
- `/robot_io/safe_joint_command_json` (`std_msgs/String`)
- `/robot_io/safety_state` (`std_msgs/String`)

### Mapping YAML

Example mapping file:

- `robotics/ros2_bridge/joint_mapping.example.yaml`

Config supports:

- `watchdog_timeout_sec`
- `default_limits` (position/velocity/effort clamps)
- per-joint mapping:
  - `input` (frontend joint id, e.g. `leg-0`)
  - `output_name`, `output_id`
  - `sign`, `scale`, `position_offset`
  - optional per-joint `limits`

### Start safety node

```bash
python robotics/ros2_bridge/safety_controller_node.py --mapping robotics/ros2_bridge/joint_mapping.example.yaml
```

### Recommended chain

`frontend -> bridge_node.py -> safety_controller_node.py -> hardware driver node`

## Driver Adapter Node (protocol conversion + smoothing + heartbeat failover)

Added node:

- `robotics/ros2_bridge/driver_adapter_node.py`

Reads safe command stream:

- `/robot_io/safe_joint_targets` (`sensor_msgs/JointState`)

Converts and publishes:

- `/driver_adapter/ros2_control/command` (`std_msgs/Float64MultiArray`)
- `/driver_adapter/can_frame_json` (`std_msgs/String`)
- `/driver_adapter/uart_packet` (`std_msgs/String`)

Smoothing:

- rate limiter (position slew)
- jerk-like limiter (acceleration step clamp)

Config:

- `robotics/ros2_bridge/driver_adapter.example.yaml`
  - `link.protocol`: `ros2_control | can | uart`
  - `link.heartbeat_timeout_sec`
  - `smoothing.max_velocity_step`
  - `smoothing.max_acceleration_step`

Run:

```bash
python robotics/ros2_bridge/driver_adapter_node.py --config robotics/ros2_bridge/driver_adapter.example.yaml
```

## Hardware Heartbeat Protocol

Bridge publishes ping:

- `/robot_io/driver_heartbeat_ping` (`std_msgs/String`)
  - payload: `{"type":"driver-heartbeat-ping","seq":N,"ts":...}`

Driver adapter responds:

- `/robot_io/driver_heartbeat_ack` (`std_msgs/String`)
  - payload: `{"type":"driver-heartbeat-ack","seq":N,"ts":...}`

Bridge exposes heartbeat in WebSocket sensor state:

- `sensor-state.state.driverHeartbeat`

Driver adapter failover:

- if no ping within `heartbeat_timeout_sec`, auto-publishes zero commands (safe stop)

## Joint Calibration Workflow (auto zero/direction/limits -> YAML)

Added utility:

- `robotics/ros2_bridge/calibration_workflow.py`

Inputs:

- measured joints: `/joint_states`
- commanded safe targets: `/robot_io/safe_joint_targets`

What it updates:

- `position_offset` (zero-offset auto estimation)
- `sign` (direction check from command/measure correlation)
- `limits.position_min/max` (observed envelope with margin)

Run:

```bash
python robotics/ros2_bridge/calibration_workflow.py \
  --mapping robotics/ros2_bridge/joint_mapping.example.yaml \
  --output robotics/ros2_bridge/joint_mapping.calibrated.yaml \
  --samples 240
```

Recommended startup order:

1. `bridge_node.py`
2. `safety_controller_node.py`
3. `driver_adapter_node.py`
4. `calibration_workflow.py` (when doing calibration session)

## Notes

- This is a bridge layer, not a hardware driver.
- Real robot control loop and final interlocks should be implemented in hardware driver/controller.
- Safety controller here is a practical first guard layer, not a certified safety system.

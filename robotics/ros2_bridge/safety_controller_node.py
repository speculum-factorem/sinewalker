#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String


@dataclass
class JointLimits:
    position_min: float
    position_max: float
    velocity_abs_max: float
    effort_abs_max: float


@dataclass
class JointMap:
    input_name: str
    output_name: str
    output_id: int
    sign: float
    scale: float
    position_offset: float
    limits: JointLimits


class SafetyController(Node):
    def __init__(self, mapping_path: Path):
        super().__init__("robot_io_safety_controller")
        self._mapping_path = mapping_path
        self._watchdog_timeout = 0.35
        self._joint_maps: Dict[str, JointMap] = {}
        self._estop = False
        self._last_cmd_time = 0.0

        self._load_config(mapping_path)

        self._sub_cmd_json = self.create_subscription(String, "/robot_io/joint_command_json", self._on_command_json, 20)
        self._sub_estop = self.create_subscription(Bool, "/robot_io/e_stop", self._on_estop, 10)

        self._pub_safe_json = self.create_publisher(String, "/robot_io/safe_joint_command_json", 20)
        self._pub_safe_pos = self.create_publisher(Float64MultiArray, "/robot_io/safe_joint_target_position", 20)
        self._pub_safe_vel = self.create_publisher(Float64MultiArray, "/robot_io/safe_joint_target_velocity", 20)
        self._pub_safe_eff = self.create_publisher(Float64MultiArray, "/robot_io/safe_joint_target_effort", 20)
        self._pub_safe_joint_state = self.create_publisher(JointState, "/robot_io/safe_joint_targets", 20)
        self._pub_safety_state = self.create_publisher(String, "/robot_io/safety_state", 10)

        self.create_timer(0.05, self._watchdog_tick)

    def _load_config(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError("Invalid mapping yaml: root must be a mapping")

        self._watchdog_timeout = float(raw.get("watchdog_timeout_sec", 0.35))
        default_limits_raw = raw.get("default_limits", {})
        default_limits = JointLimits(
            position_min=float(default_limits_raw.get("position_min", -1.2)),
            position_max=float(default_limits_raw.get("position_max", 1.2)),
            velocity_abs_max=float(default_limits_raw.get("velocity_abs_max", 4.0)),
            effort_abs_max=float(default_limits_raw.get("effort_abs_max", 18.0)),
        )

        joints = raw.get("joints", [])
        if not isinstance(joints, list) or not joints:
            raise RuntimeError("Invalid mapping yaml: joints must be a non-empty list")

        parsed: Dict[str, JointMap] = {}
        for item in joints:
            if not isinstance(item, dict):
                continue
            inp = str(item.get("input", "")).strip()
            if not inp:
                continue
            lim_raw = item.get("limits", {}) if isinstance(item.get("limits", {}), dict) else {}
            limits = JointLimits(
                position_min=float(lim_raw.get("position_min", default_limits.position_min)),
                position_max=float(lim_raw.get("position_max", default_limits.position_max)),
                velocity_abs_max=float(lim_raw.get("velocity_abs_max", default_limits.velocity_abs_max)),
                effort_abs_max=float(lim_raw.get("effort_abs_max", default_limits.effort_abs_max)),
            )
            parsed[inp] = JointMap(
                input_name=inp,
                output_name=str(item.get("output_name", inp)),
                output_id=int(item.get("output_id", 0)),
                sign=float(item.get("sign", 1.0)),
                scale=float(item.get("scale", 1.0)),
                position_offset=float(item.get("position_offset", 0.0)),
                limits=limits,
            )

        if not parsed:
            raise RuntimeError("No valid joint mappings were parsed")
        self._joint_maps = parsed
        self.get_logger().info(f"Loaded {len(self._joint_maps)} joint mappings from {path}")

    def _on_estop(self, msg: Bool) -> None:
        self._estop = bool(msg.data)
        self._publish_safety_state("estop" if self._estop else "ok")

    def _on_command_json(self, msg: String) -> None:
        now = time.monotonic()
        self._last_cmd_time = now
        if self._estop:
            self._publish_zero_commands(reason="estop")
            return

        try:
            payload = json.loads(msg.data)
            commands = payload.get("commands", [])
            if not isinstance(commands, list):
                raise ValueError("commands must be a list")
            self._publish_safe(commands)
        except Exception as exc:
            self.get_logger().warn(f"Invalid command payload: {exc}")
            self._publish_zero_commands(reason="invalid-payload")

    def _watchdog_tick(self) -> None:
        if self._estop:
            return
        now = time.monotonic()
        if self._last_cmd_time <= 0:
            return
        if now - self._last_cmd_time > self._watchdog_timeout:
            self._publish_zero_commands(reason="watchdog-timeout")
            self._last_cmd_time = now

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _publish_safe(self, commands: List[Dict[str, Any]]) -> None:
        names: List[str] = []
        positions: List[float] = []
        velocities: List[float] = []
        efforts: List[float] = []
        safe_commands: List[Dict[str, Any]] = []

        for c in commands:
            if not isinstance(c, dict):
                continue
            input_name = str(c.get("jointId", "")).strip()
            if input_name not in self._joint_maps:
                continue
            m = self._joint_maps[input_name]

            raw_pos = float(c.get("targetPosition", 0.0))
            raw_vel = float(c.get("targetVelocity", 0.0))
            raw_eff = float(c.get("targetEffort", 0.0))

            pos = m.sign * m.scale * raw_pos + m.position_offset
            vel = m.sign * m.scale * raw_vel
            eff = m.sign * m.scale * raw_eff

            pos = self._clamp(pos, m.limits.position_min, m.limits.position_max)
            vel = self._clamp(vel, -m.limits.velocity_abs_max, m.limits.velocity_abs_max)
            eff = self._clamp(eff, -m.limits.effort_abs_max, m.limits.effort_abs_max)

            names.append(m.output_name)
            positions.append(pos)
            velocities.append(vel)
            efforts.append(eff)
            safe_commands.append(
                {
                    "outputName": m.output_name,
                    "outputId": m.output_id,
                    "targetPosition": pos,
                    "targetVelocity": vel,
                    "targetEffort": eff,
                    "sourceJoint": m.input_name,
                }
            )

        self._publish_all(names, positions, velocities, efforts, safe_commands, reason="ok")

    def _publish_zero_commands(self, reason: str) -> None:
        names = [m.output_name for m in self._joint_maps.values()]
        positions = [0.0] * len(names)
        velocities = [0.0] * len(names)
        efforts = [0.0] * len(names)
        safe_commands = [
            {
                "outputName": m.output_name,
                "outputId": m.output_id,
                "targetPosition": 0.0,
                "targetVelocity": 0.0,
                "targetEffort": 0.0,
                "sourceJoint": m.input_name,
            }
            for m in self._joint_maps.values()
        ]
        self._publish_all(names, positions, velocities, efforts, safe_commands, reason=reason)

    def _publish_all(
        self,
        names: List[str],
        positions: List[float],
        velocities: List[float],
        efforts: List[float],
        safe_commands: List[Dict[str, Any]],
        reason: str,
    ) -> None:
        pos_msg = Float64MultiArray()
        vel_msg = Float64MultiArray()
        eff_msg = Float64MultiArray()
        pos_msg.data = positions
        vel_msg.data = velocities
        eff_msg.data = efforts
        self._pub_safe_pos.publish(pos_msg)
        self._pub_safe_vel.publish(vel_msg)
        self._pub_safe_eff.publish(eff_msg)

        js = JointState()
        js.name = names
        js.position = positions
        js.velocity = velocities
        js.effort = efforts
        self._pub_safe_joint_state.publish(js)

        s = String()
        s.data = json.dumps({"type": "safe_joint_command", "reason": reason, "commands": safe_commands}, ensure_ascii=False)
        self._pub_safe_json.publish(s)
        self._publish_safety_state(reason)

    def _publish_safety_state(self, reason: str) -> None:
        s = String()
        s.data = json.dumps(
            {
                "estop": self._estop,
                "watchdogTimeoutSec": self._watchdog_timeout,
                "reason": reason,
                "mappedJointCount": len(self._joint_maps),
            },
            ensure_ascii=False,
        )
        self._pub_safety_state.publish(s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot I/O safety controller node")
    parser.add_argument(
        "--mapping",
        default="robotics/ros2_bridge/joint_mapping.example.yaml",
        help="Path to joint mapping yaml",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping).resolve()
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    rclpy.init()
    node = SafetyController(mapping_path)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

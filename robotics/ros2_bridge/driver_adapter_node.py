#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String


@dataclass
class SmoothingConfig:
    max_velocity_step: float
    max_acceleration_step: float


@dataclass
class LinkConfig:
    protocol: str
    heartbeat_timeout_sec: float


class DriverAdapter(Node):
    def __init__(self, config_path: Path):
        super().__init__("robot_driver_adapter")
        self.smoothing = SmoothingConfig(max_velocity_step=0.08, max_acceleration_step=0.12)
        self.link = LinkConfig(protocol="ros2_control", heartbeat_timeout_sec=0.8)
        self._load_config(config_path)

        self.last_ping_ts = 0.0
        self.last_ack_seq = -1
        self.prev_pos: List[float] = []
        self.prev_vel: List[float] = []
        self.last_tick_ts = 0.0
        self.joint_names: List[str] = []

        self.sub_safe = self.create_subscription(JointState, "/robot_io/safe_joint_targets", self._on_safe_targets, 20)
        self.sub_ping = self.create_subscription(String, "/robot_io/driver_heartbeat_ping", self._on_ping, 20)

        self.pub_ros2_control = self.create_publisher(Float64MultiArray, "/driver_adapter/ros2_control/command", 20)
        self.pub_can = self.create_publisher(String, "/driver_adapter/can_frame_json", 20)
        self.pub_uart = self.create_publisher(String, "/driver_adapter/uart_packet", 20)
        self.pub_ack = self.create_publisher(String, "/robot_io/driver_heartbeat_ack", 20)
        self.pub_link_state = self.create_publisher(String, "/robot_io/driver_link_state", 10)

        self.create_timer(0.05, self._heartbeat_watchdog_tick)

    def _load_config(self, config_path: Path) -> None:
        if not config_path.exists():
            self.get_logger().warn(f"Driver config not found, defaults will be used: {config_path}")
            return
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        sm = raw.get("smoothing", {})
        lk = raw.get("link", {})
        self.smoothing = SmoothingConfig(
            max_velocity_step=float(sm.get("max_velocity_step", self.smoothing.max_velocity_step)),
            max_acceleration_step=float(sm.get("max_acceleration_step", self.smoothing.max_acceleration_step)),
        )
        self.link = LinkConfig(
            protocol=str(lk.get("protocol", self.link.protocol)),
            heartbeat_timeout_sec=float(lk.get("heartbeat_timeout_sec", self.link.heartbeat_timeout_sec)),
        )

    def _on_ping(self, msg: String) -> None:
        now = time.monotonic()
        self.last_ping_ts = now
        seq = -1
        try:
            payload = json.loads(msg.data)
            seq = int(payload.get("seq", -1))
        except Exception:
            pass
        self.last_ack_seq = seq
        ack = String()
        ack.data = json.dumps({"type": "driver-heartbeat-ack", "seq": seq, "ts": int(time.time() * 1000)}, ensure_ascii=False)
        self.pub_ack.publish(ack)
        self._publish_link_state("ok")

    def _heartbeat_watchdog_tick(self) -> None:
        if self.last_ping_ts <= 0:
            return
        elapsed = time.monotonic() - self.last_ping_ts
        if elapsed > self.link.heartbeat_timeout_sec:
            self._publish_zero(reason="heartbeat-timeout")
            self._publish_link_state("timeout")

    def _publish_link_state(self, state: str) -> None:
        s = String()
        s.data = json.dumps(
            {
                "state": state,
                "protocol": self.link.protocol,
                "heartbeatTimeoutSec": self.link.heartbeat_timeout_sec,
                "lastAckSeq": self.last_ack_seq,
            },
            ensure_ascii=False,
        )
        self.pub_link_state.publish(s)

    def _on_safe_targets(self, msg: JointState) -> None:
        self.joint_names = list(msg.name)
        target_pos = list(msg.position)
        target_vel = list(msg.velocity) if msg.velocity else [0.0] * len(target_pos)
        target_eff = list(msg.effort) if msg.effort else [0.0] * len(target_pos)
        if not target_pos:
            return

        now = time.monotonic()
        dt = max(1e-4, now - self.last_tick_ts) if self.last_tick_ts > 0 else 0.02
        self.last_tick_ts = now

        if not self.prev_pos or len(self.prev_pos) != len(target_pos):
            self.prev_pos = target_pos[:]
            self.prev_vel = [0.0] * len(target_pos)

        out_pos: List[float] = []
        out_vel: List[float] = []
        out_eff: List[float] = []
        for i in range(len(target_pos)):
            p_prev = self.prev_pos[i]
            v_prev = self.prev_vel[i]
            p_target = target_pos[i]
            v_target = target_vel[i]

            # Rate limiter (position slew)
            max_dp = self.smoothing.max_velocity_step * dt
            dp = max(-max_dp, min(max_dp, p_target - p_prev))
            p_rate = p_prev + dp

            # Jerk-like limiter via velocity change clamp
            v_rate = (p_rate - p_prev) / dt
            max_dv = self.smoothing.max_acceleration_step * dt
            dv = max(-max_dv, min(max_dv, v_rate - v_prev))
            v_safe = v_prev + dv
            p_safe = p_prev + v_safe * dt

            out_pos.append(p_safe)
            out_vel.append(v_safe if v_target == 0.0 else v_target)
            out_eff.append(target_eff[i] if i < len(target_eff) else 0.0)

        self.prev_pos = out_pos[:]
        self.prev_vel = out_vel[:]
        self._publish_driver_commands(out_pos, out_vel, out_eff, reason="ok")

    def _publish_zero(self, reason: str) -> None:
        if not self.prev_pos:
            return
        zeros = [0.0] * len(self.prev_pos)
        self.prev_vel = [0.0] * len(self.prev_pos)
        self._publish_driver_commands(zeros, zeros, zeros, reason=reason)

    def _publish_driver_commands(self, pos: List[float], vel: List[float], eff: List[float], reason: str) -> None:
        arr = Float64MultiArray()
        arr.data = pos
        self.pub_ros2_control.publish(arr)

        can = String()
        can.data = json.dumps(
            {"type": "can-command-batch", "reason": reason, "joints": self.joint_names, "position": pos, "velocity": vel, "effort": eff},
            ensure_ascii=False,
        )
        self.pub_can.publish(can)

        uart = String()
        uart.data = json.dumps(
            {"type": "uart-command-batch", "reason": reason, "joints": self.joint_names, "position": pos, "velocity": vel, "effort": eff},
            ensure_ascii=False,
        )
        self.pub_uart.publish(uart)


def main() -> None:
    parser = argparse.ArgumentParser(description="Driver adapter node with smoothing and heartbeat watchdog")
    parser.add_argument("--config", default="robotics/ros2_bridge/driver_adapter.example.yaml")
    args = parser.parse_args()

    rclpy.init()
    node = DriverAdapter(Path(args.config).resolve())
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

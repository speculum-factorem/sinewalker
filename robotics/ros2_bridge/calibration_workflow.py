#!/usr/bin/env python3
import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState


@dataclass
class JointSample:
    measured: List[float] = field(default_factory=list)
    commanded: List[float] = field(default_factory=list)


class CalibrationWorkflow(Node):
    def __init__(self, mapping_path: Path, output_path: Path, sample_count: int):
        super().__init__("joint_calibration_workflow")
        self.mapping_path = mapping_path
        self.output_path = output_path
        self.sample_count = sample_count
        self.mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
        self.joint_by_output = {j.get("output_name"): j for j in self.mapping.get("joints", []) if isinstance(j, dict)}
        self.samples: Dict[str, JointSample] = {name: JointSample() for name in self.joint_by_output.keys()}
        self.sub_measured = self.create_subscription(JointState, "/joint_states", self.on_joint_states, 20)
        self.sub_commanded = self.create_subscription(JointState, "/robot_io/safe_joint_targets", self.on_commanded, 20)
        self.done = False

    def on_joint_states(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name in self.samples and i < len(msg.position):
                self.samples[name].measured.append(float(msg.position[i]))
        self.check_done()

    def on_commanded(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name in self.samples and i < len(msg.position):
                self.samples[name].commanded.append(float(msg.position[i]))
        self.check_done()

    def check_done(self) -> None:
        if self.done:
            return
        ok = all(len(s.measured) >= self.sample_count and len(s.commanded) >= self.sample_count for s in self.samples.values())
        if ok:
            self.done = True
            self.compute_and_save()

    def compute_and_save(self) -> None:
        for name, s in self.samples.items():
            joint = self.joint_by_output[name]
            measured = s.measured[: self.sample_count]
            commanded = s.commanded[: self.sample_count]
            if not measured:
                continue

            zero_offset = -statistics.mean(measured)
            joint["position_offset"] = float(joint.get("position_offset", 0.0)) + zero_offset

            direction_sign = self.estimate_direction_sign(commanded, measured)
            joint["sign"] = float(joint.get("sign", 1.0)) * direction_sign

            lo = min(measured) - 0.05
            hi = max(measured) + 0.05
            limits = joint.get("limits", {})
            if not isinstance(limits, dict):
                limits = {}
            limits["position_min"] = float(min(limits.get("position_min", lo), lo))
            limits["position_max"] = float(max(limits.get("position_max", hi), hi))
            joint["limits"] = limits

        out_text = yaml.safe_dump(self.mapping, sort_keys=False, allow_unicode=True)
        self.output_path.write_text(out_text, encoding="utf-8")
        summary = {
            "type": "calibration-result",
            "samples": self.sample_count,
            "output": str(self.output_path),
            "joints": list(self.samples.keys()),
        }
        self.get_logger().info(json.dumps(summary, ensure_ascii=False))

    @staticmethod
    def estimate_direction_sign(commanded: List[float], measured: List[float]) -> float:
        n = min(len(commanded), len(measured))
        if n < 3:
            return 1.0
        dc = [commanded[i] - commanded[i - 1] for i in range(1, n)]
        dm = [measured[i] - measured[i - 1] for i in range(1, n)]
        score = sum(dc[i] * dm[i] for i in range(min(len(dc), len(dm))))
        return 1.0 if score >= 0 else -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto calibration workflow for joint mapping yaml")
    parser.add_argument("--mapping", default="robotics/ros2_bridge/joint_mapping.example.yaml")
    parser.add_argument("--output", default="robotics/ros2_bridge/joint_mapping.calibrated.yaml")
    parser.add_argument("--samples", type=int, default=240)
    args = parser.parse_args()

    mapping_path = Path(args.mapping).resolve()
    output_path = Path(args.output).resolve()

    rclpy.init()
    node = CalibrationWorkflow(mapping_path, output_path, max(40, args.samples))
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

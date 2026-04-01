#!/usr/bin/env python3
import argparse
import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray, String
from websockets.server import WebSocketServerProtocol, serve


@dataclass
class SharedState:
    clients: Set[WebSocketServerProtocol] = field(default_factory=set)
    latest_imu: Optional[Dict[str, Any]] = None
    latest_joint_state: Optional[Dict[str, Any]] = None
    latest_twist: Optional[Dict[str, Any]] = None


class Ros2RobotBridge(Node):
    def __init__(self, state: SharedState):
        super().__init__("robot_io_bridge")
        self.state = state

        self.pub_joint_pos = self.create_publisher(Float64MultiArray, "/robot_io/joint_target_position", 10)
        self.pub_joint_vel = self.create_publisher(Float64MultiArray, "/robot_io/joint_target_velocity", 10)
        self.pub_joint_eff = self.create_publisher(Float64MultiArray, "/robot_io/joint_target_effort", 10)
        self.pub_joint_json = self.create_publisher(String, "/robot_io/joint_command_json", 10)

        self.create_subscription(Imu, "/imu/data", self.on_imu, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 10)
        self.create_subscription(Twist, "/robot/velocity", self.on_twist, 10)
        self.create_subscription(String, "/robot_io/driver_heartbeat_ack", self.on_driver_ack, 10)

        self.pub_driver_ping = self.create_publisher(String, "/robot_io/driver_heartbeat_ping", 10)
        self.driver_last_ack_ts = 0
        self.driver_last_ack_seq = -1
        self.driver_ping_seq = 0
        self.create_timer(0.2, self.publish_driver_ping)

    def publish_motor_commands(self, commands: List[Dict[str, Any]], raw_payload: Dict[str, Any]) -> None:
        pos_msg = Float64MultiArray()
        vel_msg = Float64MultiArray()
        eff_msg = Float64MultiArray()
        pos_msg.data = [float(c.get("targetPosition", 0.0)) for c in commands]
        vel_msg.data = [float(c.get("targetVelocity", 0.0)) for c in commands]
        eff_msg.data = [float(c.get("targetEffort", 0.0)) for c in commands]
        self.pub_joint_pos.publish(pos_msg)
        self.pub_joint_vel.publish(vel_msg)
        self.pub_joint_eff.publish(eff_msg)

        js = String()
        js.data = json.dumps(
            {
                "type": "joint_command",
                "count": len(commands),
                "commands": commands,
                "sourceTs": raw_payload.get("ts"),
                "sourceTimeSeconds": raw_payload.get("timeSeconds"),
            },
            ensure_ascii=False,
        )
        self.pub_joint_json.publish(js)

    def on_imu(self, msg: Imu) -> None:
        self.state.latest_imu = {
            "orientation": {
                "x": msg.orientation.x,
                "y": msg.orientation.y,
                "z": msg.orientation.z,
                "w": msg.orientation.w,
            },
            "angularVelocity": {
                "x": msg.angular_velocity.x,
                "y": msg.angular_velocity.y,
                "z": msg.angular_velocity.z,
            },
            "linearAcceleration": {
                "x": msg.linear_acceleration.x,
                "y": msg.linear_acceleration.y,
                "z": msg.linear_acceleration.z,
            },
        }

    def on_joint_state(self, msg: JointState) -> None:
        self.state.latest_joint_state = {
            "name": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "effort": list(msg.effort),
        }

    def on_twist(self, msg: Twist) -> None:
        self.state.latest_twist = {
            "linear": {"x": msg.linear.x, "y": msg.linear.y, "z": msg.linear.z},
            "angular": {"x": msg.angular.x, "y": msg.angular.y, "z": msg.angular.z},
        }

    def publish_driver_ping(self) -> None:
        self.driver_ping_seq += 1
        msg = String()
        msg.data = json.dumps(
            {
                "type": "driver-heartbeat-ping",
                "seq": self.driver_ping_seq,
                "ts": int(time.time() * 1000),
            },
            ensure_ascii=False,
        )
        self.pub_driver_ping.publish(msg)

    def on_driver_ack(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self.driver_last_ack_seq = int(payload.get("seq", -1))
        except Exception:
            self.driver_last_ack_seq = -1
        self.driver_last_ack_ts = int(time.time() * 1000)


class WsServer:
    def __init__(self, ros_bridge: Ros2RobotBridge, state: SharedState, host: str, port: int):
        self.ros_bridge = ros_bridge
        self.state = state
        self.host = host
        self.port = port

    async def ws_handler(self, websocket: WebSocketServerProtocol) -> None:
        self.state.clients.add(websocket)
        self.ros_bridge.get_logger().info(f"WS client connected: {websocket.remote_address}")
        try:
            async for raw in websocket:
                await self.handle_message(websocket, raw)
        finally:
            self.state.clients.discard(websocket)
            self.ros_bridge.get_logger().info(f"WS client disconnected: {websocket.remote_address}")

    async def handle_message(self, websocket: WebSocketServerProtocol, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "error", "message": "invalid-json"}))
            return

        msg_type = payload.get("type")
        if msg_type == "robot-io-tick":
            commands = payload.get("commands", [])
            if not isinstance(commands, list):
                await websocket.send(json.dumps({"type": "error", "message": "commands-must-be-list"}))
                return
            self.ros_bridge.publish_motor_commands(commands, payload)
            await websocket.send(
                json.dumps(
                    {
                        "type": "ack",
                        "receivedCommands": len(commands),
                        "sensorState": self.current_sensor_state(),
                    }
                )
            )
            return

        if msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
            return

        await websocket.send(json.dumps({"type": "error", "message": f"unknown-type:{msg_type}"}))

    def current_sensor_state(self) -> Dict[str, Any]:
        return {
            "imu": self.state.latest_imu,
            "jointState": self.state.latest_joint_state,
            "twist": self.state.latest_twist,
            "driverHeartbeat": {
                "lastAckTs": self.ros_bridge.driver_last_ack_ts,
                "lastAckSeq": self.ros_bridge.driver_last_ack_seq,
                "lastPingSeq": self.ros_bridge.driver_ping_seq,
            },
        }

    async def sensor_broadcast_loop(self) -> None:
        while True:
            if self.state.clients:
                msg = json.dumps(
                    {
                        "type": "sensor-state",
                        "state": self.current_sensor_state(),
                    }
                )
                dead: List[WebSocketServerProtocol] = []
                for client in self.state.clients:
                    try:
                        await client.send(msg)
                    except Exception:
                        dead.append(client)
                for d in dead:
                    self.state.clients.discard(d)
            await asyncio.sleep(0.05)

    async def run(self) -> None:
        async with serve(self.ws_handler, self.host, self.port):
            self.ros_bridge.get_logger().info(f"WebSocket bridge on ws://{self.host}:{self.port}")
            await self.sensor_broadcast_loop()


def start_ws_thread(bridge: Ros2RobotBridge, state: SharedState, host: str, port: int) -> threading.Thread:
    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server = WsServer(bridge, state, host, port)
        loop.run_until_complete(server.run())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 <-> WebSocket Robot I/O bridge")
    parser.add_argument("--ws-host", default="0.0.0.0")
    parser.add_argument("--ws-port", default=8765, type=int)
    args = parser.parse_args()

    rclpy.init()
    state = SharedState()
    bridge = Ros2RobotBridge(state)
    start_ws_thread(bridge, state, args.ws_host, args.ws_port)

    try:
        rclpy.spin(bridge)
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

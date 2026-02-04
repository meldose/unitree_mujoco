#!/usr/bin/env python3
import argparse
import json
import socket
import time

import pygame


def _parse_args():
    p = argparse.ArgumentParser(description="Xbox gamepad -> UDP sender (Windows).")
    p.add_argument("--host", required=True, help="WSL IP address.")
    p.add_argument("--port", type=int, default=49555, help="UDP port.")
    p.add_argument("--rate", type=float, default=100.0, help="Send rate (Hz).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() <= 0:
        print("No gamepad detected.")
        return 1

    js = pygame.joystick.Joystick(0)
    js.init()

    axis_id = {
        "LX": 0,
        "LY": 1,
        "RX": 3,
        "RY": 4,
        "LT": 2,
        "RT": 5,
        "DX": 6,
        "DY": 7,
    }
    button_id = {
        "X": 2,
        "Y": 3,
        "B": 1,
        "A": 0,
        "LB": 4,
        "RB": 5,
        "SELECT": 6,
        "START": 7,
    }

    key_map = {
        "R1": 0,
        "L1": 1,
        "start": 2,
        "select": 3,
        "R2": 4,
        "L2": 5,
        "F1": 6,
        "F2": 7,
        "A": 8,
        "B": 9,
        "X": 10,
        "Y": 11,
        "up": 12,
        "right": 13,
        "down": 14,
        "left": 15,
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / max(args.rate, 1.0)

    while True:
        pygame.event.get()

        key_state = [0] * 16
        key_state[key_map["R1"]] = js.get_button(button_id["RB"])
        key_state[key_map["L1"]] = js.get_button(button_id["LB"])
        key_state[key_map["start"]] = js.get_button(button_id["START"])
        key_state[key_map["select"]] = js.get_button(button_id["SELECT"])
        key_state[key_map["R2"]] = int(js.get_axis(axis_id["RT"]) > 0)
        key_state[key_map["L2"]] = int(js.get_axis(axis_id["LT"]) > 0)
        key_state[key_map["A"]] = js.get_button(button_id["A"])
        key_state[key_map["B"]] = js.get_button(button_id["B"])
        key_state[key_map["X"]] = js.get_button(button_id["X"])
        key_state[key_map["Y"]] = js.get_button(button_id["Y"])
        # D-pad via hat
        hat_x, hat_y = js.get_hat(0)
        key_state[key_map["up"]] = int(hat_y > 0)
        key_state[key_map["right"]] = int(hat_x > 0)
        key_state[key_map["down"]] = int(hat_y < 0)
        key_state[key_map["left"]] = int(hat_x < 0)

        key_value = 0
        for i in range(16):
            key_value |= (int(key_state[i]) & 1) << i

        payload = {
            "keys": int(key_value),
            "lx": float(js.get_axis(axis_id["LX"])),
            "ly": float(-js.get_axis(axis_id["LY"])),
            "rx": float(js.get_axis(axis_id["RX"])),
            "ry": float(-js.get_axis(axis_id["RY"])),
        }

        sock.sendto(json.dumps(payload).encode("utf-8"), (args.host, args.port))
        time.sleep(period)


if __name__ == "__main__":
    raise SystemExit(main())

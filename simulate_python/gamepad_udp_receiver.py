#!/home/meldose/.pyenv/versions/unitree310/bin/python
import argparse
import json
import socket
import time
from pathlib import Path
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_

# Ensure we import the local simulate_python/config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"


def _parse_args():
    p = argparse.ArgumentParser(description="UDP -> WirelessController bridge (WSL receiver).")
    p.add_argument("--bind", default="0.0.0.0", help="Bind address.")
    p.add_argument("--port", type=int, default=49555, help="UDP port.")
    p.add_argument("--rate", type=float, default=100.0, help="Publish rate (Hz).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    pub = ChannelPublisher(TOPIC_WIRELESS_CONTROLLER, WirelessController_)
    pub.Init()
    msg = unitree_go_msg_dds__WirelessController_()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.setblocking(False)

    last = time.time()
    period = 1.0 / max(args.rate, 1.0)

    while True:
        now = time.time()
        # Consume all queued packets, keep last
        try:
            while True:
                data, _addr = sock.recvfrom(2048)
                payload = json.loads(data.decode("utf-8"))
                msg.keys = int(payload.get("keys", 0))
                msg.lx = float(payload.get("lx", 0.0))
                msg.ly = float(payload.get("ly", 0.0))
                msg.rx = float(payload.get("rx", 0.0))
                msg.ry = float(payload.get("ry", 0.0))
        except BlockingIOError:
            pass

        if now - last >= period:
            pub.Write(msg)
            last = now
        else:
            time.sleep(0.001)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

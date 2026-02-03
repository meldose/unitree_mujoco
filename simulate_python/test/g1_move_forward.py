#!/home/meldose/.pyenv/versions/unitree310/bin/python
import argparse
import sys
import time
from pathlib import Path
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

# Ensure we import the local simulate_python/config.py, even if this file is copied.
def _add_simulate_python_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "config.py"
        if candidate.exists():
            sys.path.insert(0, str(parent))
            return
    # Fallback to original simulate_python path if present
    fallback = Path("/home/meldose/g1/unitree_mujoco/simulate_python")
    if (fallback / "config.py").exists():
        sys.path.insert(0, str(fallback))


_add_simulate_python_to_path()
import config


def main() -> int:
    parser = argparse.ArgumentParser(description="Move G1 forward in simulation.")
    parser.add_argument("--vx", type=float, default=0.3, help="Forward velocity (m/s)")
    parser.add_argument("--duration", type=float, default=3.0, help="Move time (s). Ignored with --continuous")
    parser.add_argument("--stand", choices=["high", "low", "none"], default="high")
    parser.add_argument("--continuous", action="store_true", help="Move until Ctrl+C")
    args = parser.parse_args()

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    client = LocoClient()
    client.SetTimeout(10.0)
    client.Init()

    # Ensure a stable starting posture.
    client.Damp()
    time.sleep(0.5)
    if args.stand == "high":
        client.HighStand()
        time.sleep(0.5)
    elif args.stand == "low":
        client.LowStand()
        time.sleep(0.5)

    print(f"[g1_move_forward] Moving forward vx={args.vx} m/s")
    start = time.time()
    if args.continuous:
        try:
            while True:
                client.Move(args.vx, 0.0, 0.0)
                print(f"[g1_move_forward] t={time.time() - start:.2f}s command sent")
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("[g1_move_forward] Stopping (Ctrl+C)")
    else:
        client.Move(args.vx, 0.0, 0.0)
        print(f"[g1_move_forward] Holding for {args.duration:.2f}s")
        time.sleep(args.duration)

    # Stop motion and relax.
    client.Move(0.0, 0.0, 0.0)
    print("[g1_move_forward] Stop command sent")
    time.sleep(0.2)
    client.Damp()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

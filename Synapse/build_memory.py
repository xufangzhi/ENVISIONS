import argparse
import os
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["miniwob", "mind2web"])
    parser.add_argument("--mind2web_data_dir", type=str)
    parser.add_argument("--mind2web_top_k_elements", type=int, default=3)
    args = parser.parse_args()

    current_path = os.getcwd()
    if args.env == "miniwob":
        from synapse.memory.miniwob.build_memory import build_memory

        memory_path = os.path.join(current_path, "synapse/memory/miniwob")
        build_memory(memory_path)
    else:
        from synapse.memory.mind2web.build_memory import build_memory

        memory_path = os.path.join(current_path, "synapse/memory/mind2web")
        log_dir = Path(memory_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        build_memory(memory_path, args.mind2web_data_dir, args.mind2web_top_k_elements)

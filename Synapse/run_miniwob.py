import argparse
import logging
import os

from synapse.agents.miniwob import Agent

logger = logging.getLogger("synapse")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.propagate = False


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--env_name", type=str, default="simple-algebra")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_filter", action="store_true", default=False)
    parser.add_argument("--no_memory", action="store_true", default=True)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    current_path = os.getcwd()
    args.memory_path = os.path.join(current_path, "synapse/memory/miniwob")
    args.log_dir = os.path.join(current_path, "results/miniwob")

    agent = Agent(args=args)
    if args.env_name in ["book-flight", "terminal", "use-autocomplete"]:
        max_steps = 2
    elif args.env_name in ["login-user", "login-user-popup"]:
        max_steps = 3
    elif args.env_name in ["guess-number", "tic-tac-toe"]:
        max_steps = 10
    else:
        max_steps = 1
    for i in range(args.num_episodes):
        args.seed = 5
        print(11111)
        agent.reset(seed=args.seed + i)
        print(111)
        for _ in range(max_steps):
            obs = agent.filter()
            actions = agent.act(obs)
            print(obs)
            print("==========")
            if actions is None:
                break
            try:
                logger.info(f"Actions:\n{actions}")
                exec(actions)
            except:
                logger.info(f"Failed to execute action. Try again.")
            if agent.done:
                break
        agent.log_results()
    agent.close()


if __name__ == "__main__":
    main()
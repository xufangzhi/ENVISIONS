import argparse
import logging
import os
import json
import random
import time
from tqdm import tqdm
import re

# import Synapse.synapse.utils.llm
os.chdir("symbol-llm-v2/Synapse")
from .synapse.agents.miniwob_seeclick import Agent


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--env_name", type=str, default="click-collapsible")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--part_id",type=int,default=1,help="part_id to determine the driver",)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_filter", action="store_true", default=True)
    parser.add_argument("--no_memory", action="store_true", default=True)

    return parser


def extract_seed_from_string(s):
    # Use regular expression to find the number after 'seed'
    match = re.search(r'seed(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        return None


def main():
    parser = create_parser()
    args = parser.parse_args()

    envs_all = os.listdir('./synapse_rollout')

    envs_num = 0
    print(len(envs_all))

    for envs in tqdm(envs_all):
        if envs == '.DS_Store':
            continue

        print("Env: " + envs)
        args.env_name = envs
        agent = Agent(args=args)

        envs_num += 1
        for i in range(args.num_episodes):
            data_dict = {}   # contain reconstructed train sample
            data_dict['env_name'] = envs
            data_dict['seed'] = str(i+args.seed)
            # obs = agent.reset(seed=args.seed + i)
            # print(obs)
            print("Seed: " + str(i+args.seed))

            try:
                agent.reset(seed=int(i+args.seed))
                print(1111)
            except:
                continue
            # print(actions)
            # print(action)
            # print(obs)
            # try:
            #     exec(action)  # 执行actions的同时会在agent.record_traj中记录
            #     if agent.reward > 0.8:
            #         # print("success")
            #         logging.info(f'success')
            #         preference_scores_each_sample[j] = 1
            # except:
            #     continue

            agent.close()

if __name__ == "__main__":
    main()

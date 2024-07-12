import os
os.chdir("symbol-llm-v2/Synapse/")

import sys
sys.path.append(os.getcwd())

import json
import re

import numpy as np
import time
import signal

import argparse
import logging

# import synapse.utils.llm
from synapse.agents.miniwob_seeclick import Agent

# logger = logging.getLogger('self_training_logger')
# logger.setLevel(logging.DEBUG)



def parse_args():
    parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
    parser.add_argument(
        "--domain",
        type=str,
        default="agent",
        help="The name of the domain [math,agent,logic].",
    )
    parser.add_argument(
        "--cur_iter",
        type=int,
        default=1,
        help="The index of the current iteration",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="miniwob_v17_deepseekchat",
        help="The prefix for the dataset name, directory name and save path",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseekchat",
        help="Base model",
    )
    parser.add_argument(
        "--part_id",
        type=int,
        default=6,
        help="part id",
    )
    parser.add_argument(
        "--repaired",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Whether to use few-shot prompting",
    )
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--env_name", type=str, default="click-collapsible")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_filter", action="store_true", default=True)
    parser.add_argument("--no_memory", action="store_true", default=True)
    args = parser.parse_args()
    return args


def extract_seed_from_string(s):
    # Use regular expression to find the number after 'seed'
    match = re.search(r'seed(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_action_blocks(text):
    pattern = r"```(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    else:
        return text.strip()

def main():
    args = parse_args()
    if args.repaired:
        suffix = "_repaired"
    else:
        suffix = ""
    logging.basicConfig(filename=f'logs/output_{args.task_prefix}_part{args.part_id}_iter{args.cur_iter}{suffix}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')

    # for part_index in range(8,9):
    part = f"part{args.part_id}"

    if args.few_shot:
        with open(f"../score_memory/{args.task_prefix}/{args.task_prefix}_{part}_iter0.json") as file:
            pred = json.load(file)
    else:
        with open(f"../../new_generated_data/{args.task_prefix}_{part}_iter{args.cur_iter+1}{suffix}.json") as file:
            pred = json.load(file)

    with open(f"../open-instruct/data/miniwob_{part}.json") as file:
        gold = json.load(file)

    logging.info(f'{len(pred)}, {len(gold)}')
    # print(len(pred),len(gold))
    assert len(pred)//5==len(gold), "The length between pred and gold is not equal"

    num = 0
    preference_scores_list = []

    # for each sample
    for i in range(len(gold)):
    # for i in range(5):
        preference_scores_each_sample = [0] * 5
        args.env_name = gold[i]['env_name']
        logging.info(f"Current env: {args.env_name}")
        # print(args.env_name)
        try:
            agent = Agent(args=args)
        except:
            preference_scores_list.append(preference_scores_each_sample)
            continue

        if (i==21 and args.part_id==8) or (i==64 and args.part_id==6) or (i==67 and args.part_id==6) or (i==68 and args.part_id==6) or (i==56 and args.part_id==6) or \
                (i==249 and args.part_id==6) or (i==72 and args.part_id==6) or (i==62 and args.part_id==6) or (i==96 and args.part_id==6):
            continue
        for j in range(5):
            action = pred[i*5+j]['response']

            logging.info(f'{i}, {j}')
            print(i,j)

            # action = synapse.utils.llm.extract_from_response(action, "```")
            action = extract_action_blocks(action).strip()
            if "agent." not in action or 'state' in action or not action.startswith("agent."):
                continue
            try:
                if args.few_shot:
                    obs = agent.reset(seed=int(gold[i]['id']))
                else:
                    obs = agent.reset(seed=int(gold[i]['id']) + (args.cur_iter + 1) * 50)
            except:
                continue
            # print(actions)
            # print(action)
            # print(obs)
            try:
                exec(action)  # 执行actions的同时会在agent.record_traj中记录
                if agent.reward > 0.8:
                    # print("success")
                    logging.info(f'success')
                    preference_scores_each_sample[j] = 1
            except:
                continue
            if j==4:
                try:
                    agent.close()
                except:
                    continue
        preference_scores_list.append(preference_scores_each_sample)


    if args.few_shot:
        np.save(f"../../symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part}_iter0.npy", np.array(preference_scores_list))
    else:
        np.save(f"../../symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part}_iter{args.cur_iter+1}{suffix}.npy",np.array(preference_scores_list))

    print(num)

    os.chdir("../..")
if __name__ == "__main__":
    main()
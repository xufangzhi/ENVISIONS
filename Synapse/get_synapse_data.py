# Step1：获取synapse执行miniwob的数据
# 处理tips：
#  1.点击多个元素的处理：记录所有点击的元素，后续再按照任务处理，每个元素都点击一遍（click-shades, click-pie），除非有bbox为0（email-inbox-nl-turk)
#  2.存储了多种succeess轨迹：只使用标准Synapse的轨迹
#  3.按reward决定是否记录：reward>0.8则记录

import argparse
import logging
import os
import json
import random
from tqdm import tqdm
import re

import synapse.utils.llm
from synapse.agents.miniwob_seeclick import Agent


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--env_name", type=str, default="click-collapsible")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
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
    current_path = os.getcwd()
    args.memory_path = os.path.join(current_path, "synapse/memory/miniwob")
    args.log_dir = os.path.join(current_path, "results/miniwob")

    envs_all = os.listdir('synapse_rollout')

    traj_records = {}
    unavailable_task = ['social-media', 'social-media-all', 'choose-list', 'social-media-some', 'book-flight',
                        'click-scroll-list']

    #random.shuffle(envs_all)
    for envs in tqdm(envs_all):
        print(envs)
        if envs == '.DS_Store':
            continue
        #if not (envs == 'email-inbox-nl-turk'):
        #    continue
        if envs in unavailable_task:
            continue

        traj_records[envs] = []

        print("Env: "+envs)
        args.env_name = envs

        agent = Agent(args=args)

        # 作者提供的trajs路径
        traj_files_dir = os.path.join('synapse_rollout', args.env_name)

        if args.env_name in ["book-flight", "terminal", "use-autocomplete"]:
            max_steps = 2
        elif args.env_name in ["login-user", "login-user-popup"]:
            max_steps = 3
        elif args.env_name in ["guess-number", "tic-tac-toe"]:
            max_steps = 10
        else:
            max_steps = 1
        for i in range(args.num_episodes):
            print(i)
            obs = agent.reset(seed=args.seed + i)
            print(obs)
            print("Seed: "+str(i))
            # 加载相应seed的traj
            traj_files = os.listdir(traj_files_dir)
            for traj in traj_files:
                if ('_{}_'.format(i) in traj) and ("filt" not in traj) and ("mem" not in traj):
                    break
            traj = json.load(open(os.path.join(traj_files_dir, traj), 'r'))
            actions_all = [item["output"] for item in traj]

            for j, actions in enumerate(actions_all):
                #print("Step: "+str(j))
                actions = synapse.utils.llm.extract_from_response(actions, "```")
                if "agent" not in actions or 'state' in actions:
                    continue
                #print(actions)
                exec(actions)   # 执行actions的同时会在agent.record_traj中记录

            if agent.reward > 0.8:
                print("success")
                print(agent.reward)
                print(agent.record_traj)
                
                traj_seed = []
                for k, item in enumerate(agent.record_traj):
                    # screenshot_filename = envs+'_seed'+str(i)+'_step'+str(k)+'.jpg'
                    # screenshot_save_path = os.path.join('synapse_imgs', screenshot_filename)
                    # item["html"].save(obs)
                    #if item["action"]["action_type"] == 'click_seq':
                    #    print(item)
                    traj_step = {"html": item["html"], "action": item["action"], "goal": item["goal"]}
                    traj_seed.append(traj_step)

                traj_records[envs].append(traj_seed)
                
            agent.record_traj = []

        agent.close()

        json.dump(traj_records, open('synapse_test_goal.json', 'w'))

if __name__ == "__main__":
    main()

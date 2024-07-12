# Step1：获取synapse执行miniwob的数据
# 处理tips：
#  1.点击多个元素的处理：记录所有点击的元素，后续再按照任务处理，每个元素都点击一遍（click-shades, click-pie），除非有bbox为0（email-inbox-nl-turk)
#  2.存储了多种succeess轨迹：只使用标准Synapse的轨迹
#  3.按reward决定是否记录：reward>0.8则记录

import argparse
import collections
import logging
import os
import json
import random
import jsonlines
from tqdm import tqdm
import re

import synapse.utils.llm
from synapse.agents.miniwob_seeclick import Agent


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=300)
    parser.add_argument("--env_name", type=str, default="click-collapsible")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--part_id", type=int, default=2)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_filter", action="store_true", default=True)
    parser.add_argument("--no_memory", action="store_true", default=True)
    parser.add_argument("--append", action="store_true")

    return parser


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
        if code_blocks[0].startswith("python"):
            return code_blocks[0].split("python")[1].strip()
        else:
            return code_blocks[0].strip()
    else:
        return text.strip()

def main():
    parser = create_parser()
    args = parser.parse_args()
    current_path = os.getcwd()
    args.memory_path = os.path.join(current_path, "synapse/memory/miniwob")
    args.log_dir = os.path.join(current_path, "results/miniwob")
    # print(current_path)
    envs_all = os.listdir('./synapse_rollout')

    traj_records = {}
    # unavailable_task = ['use-autocomplete', 'book-flight', 'login-user-popup', 'login-user', 'terminal', 'click-collapsible-2']
    unavailable_envs = ['use-autocomplete', "click-menu", "click-pie", 'book-flight', 'login-user-popup', 'login-user', \
                        'terminal', 'click-collapsible-2', "email-inbox-forward-nl", "social-media-some", "email-inbox-forward-nl-turk", \
                        "email-inbox", "email-inbox-nl-turk", "tic-tac-toe", "click-tab-2", "social-media", "count-shape", \
                        "click-tab-2-hard", "find-word", "grid-coordinate"]
    # unavailable_task = ['social-media', 'social-media-all', 'choose-list', 'social-media-some', 'book-flight',
    #                     'click-scroll-list']
    # unavailable_task = []
    # print(envs_all)
    envs_num = 0
    train_data = []
    # random.shuffle(envs_all)

    # envs_all = os.listdir('./synapse_rollout')
    # envs_filter = [x for x in envs_all if x not in unavailable_envs]
    # print(sorted(envs_filter))
    # obs_dict = collections.defaultdict(list)
    #
    # if args.append:
    #     with open(f"data/miniwob_observation.json") as file:
    #         obs_dict_previous = json.load(file)
    # for envs in obs_dict_previous.keys():
    #     obs_dict[envs] = obs_dict_previous[envs]

    with open(f"data/miniwob_observation.json") as file:
        examples = json.load(file)
    print(len(examples))

    candidates = []
    with open(f"data/miniwob_claude2_iter2.jsonl") as file:
        for line in file:
            candidates.append(json.loads(line))
    print(len(candidates))

    # preference_data = []
    preference_data = []
    with open(f"../open-instruct/data/miniwob_claude2.jsonl") as file:
        for line in file:
            preference_data.append(json.loads(line))

    for i, envs in tqdm(enumerate(examples)):
        print(envs)
        if envs == '.DS_Store':
            continue

        print("Env: " + envs)
        args.env_name = envs

        try:
            agent = Agent(args=args)
        except:
            continue

        for j in range(50):
            print("cur_seed:", j+50)
            action = candidates[i * 50 + j]['response']
            action = extract_action_blocks(action).strip()
            if "agent." not in action or 'state' in action or not action.startswith("agent."):
                continue
            try:
                obs = agent.reset(seed=j+50)
            except:
                continue

            try:
                exec(action)  # 执行actions的同时会在agent.record_traj中记录
                if agent.reward > 0.8:
                    print("success")
                    # logging.info(f'success')
                    data_dict = {}
                    data_dict['envs_name'] = envs
                    data_dict['seed'] = j+50
                    instruction = "You are required to navigate the web. To accomplish the task, use methods in Agent class to generate actions, with the following functions. type(characters: str): Type a string via the keyboard. click_xpath(xpath: str): Click an HTML element with a valid XPath. press(key_type: str): Press a key on the keyboard (enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v). click_option(xpath: str): Click an option HTML element in a list with a valid XPath. movemouse(xpath: str): Move the mouse cursor on an HTML element with a valid XPath.\n"
                    data_dict['prompt'] = instruction + "\nThe observation is:\n" + obs + "\nThe action is:\n"
                    data_dict['completion'] = "```\n" + extract_action_blocks(candidates[i * 50 + j]['response']).strip() + "\n```"
                    preference_data.append(data_dict)
            except:
                continue
        try:
            agent.close()
        except:
            continue


        with jsonlines.open(f"../open-instruct/data/miniwob_claude2_iter2.jsonl", "w") as file:
            random.shuffle(preference_data)
            for i in range(len(preference_data)):
                file.write(preference_data[i])
        print(len(preference_data))

    """
    # few_shot_dict = {}
    for envs in tqdm(envs_filter):
        print(envs)
        if envs == '.DS_Store':
            continue
        # if not (envs == 'email-inbox-nl-turk'):
        #    continue
        # if envs in unavailable_task:
        #     continue

        traj_records[envs] = []

        print("Env: " + envs)
        args.env_name = envs
        #
        # agent = Agent(args=args)

        # Synapase作者提供的trajs路径
        traj_files_dir = os.path.join('synapse_rollout', args.env_name)

        envs_num += 1
        for i in range(args.num_episodes):
            data_dict = {}   # contain reconstructed train sample
            data_dict['env_name'] = envs
            data_dict['id'] = str(i)
            # obs = agent.reset(seed=args.seed + i)
            # print(obs)
            print("Seed: " + str(i))
            # 加载相应seed的traj
            traj_files = os.listdir(traj_files_dir)
            for traj in traj_files:
                if ('_{}_'.format(i) in traj) and ("filt" not in traj) and ("mem" not in traj):
                    break
            # print(traj)
            traj = json.load(open(os.path.join(traj_files_dir, traj), 'r'))[-1]

            # obtain label
            data_dict['label'] = traj['output']    # the format contains ```
            input_origin = traj['input']
            assert input_origin[-1]['role'] == "user", "mismatch"
            print(len(input_origin))
            assert len(input_origin)>=4, "length not match"
            observation = input_origin[-1]['content'].split("Observation:\n")[1].split('\nAction:')[0].strip()
            few_shot_prompt = "\nNext, two reference samples are provided.\n\n"
            for j in range(1, 3):  # from 1, skip system input
                if input_origin[j]['role'] == "user":
                    few_shot_prompt += "The observation is: " + input_origin[j]['content'].split("Observation:\n")[1].split('\nAction:')[0].strip() + "\n"
                elif input_origin[j]['role'] == "assistant":
                    few_shot_prompt += "The action is:\n" + input_origin[j]['content'] + "\n\n"
            few_shot_prompt += "Here is the test sample.\n"
            data_dict['input'] = observation
            data_dict['few_shot_prompt'] = few_shot_prompt

            # print(data_dict)
            train_data.append(data_dict)

            # few_shot_dict[envs] = few_shot_prompt

    # for i in range(4):
    #     with open(f"data/miniwob_part{i+1}.json","w") as file:
    #         json.dump(train_data[i*300:(i+1)*300], file, indent=4)
    #     print(len(train_data[i*300:(i+1)*300]))
    #
    # with open(f"data/miniwob_part5.json", "w") as file:
    #     json.dump(train_data[1200:1450], file, indent=4)
    #     print(len(train_data[1200:1450]))
    #
    # with open(f"data/miniwob_part6.json", "w") as file:
    #     json.dump(train_data[1450:1700], file, indent=4)
    #     print(len(train_data[1450:1700]))
    #
    # with open(f"data/miniwob_part7.json", "w") as file:
    #     json.dump(train_data[1700:1950], file, indent=4)
    #     print(len(train_data[1700:1950]))
    #
    # with open(f"data/miniwob_part8.json", "w") as file:
    #     json.dump(train_data[1950:], file, indent=4)
    #     print(len(train_data[1950:]))

    print(len(train_data), envs_num)

    # with open(f"data/miniwob_few_shot_prompt.json", "w") as file:
    #     json.dump(few_shot_dict, file, indent=4)
    # print(len(few_shot_dict.keys()))
    """

if __name__ == "__main__":
    main()

import collections
import json
import numpy as np
import pandas as pd
import os
import jsonlines
import random
import re
from collections import defaultdict

import re
import math
import argparse
import logging
import subprocess

logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)


def create_ngram_dict(code, n):
    ngrams = {}
    for i in range(len(code) - n + 1):
        ngram = code[i:i + n]
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
    return ngrams


def count_effective_samples(scores):
    num = 0
    for i in range(len(scores)):
        for j in range(5):
            if scores[i][j]==1:
                num += 1
                break
    return num


def extract_code_blocks(text):
    """
    :param text: original response from the model
    :return: parsed code form
    """
    text = text.strip()
    pattern = r"```(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        if code_blocks[0].startswith("python\n"):
            return "```python\n" + code_blocks[0].split("python\n")[1].strip() + "\n```"
        else:
            return "```python\n" + code_blocks[0] + "\n```"
    else:
        return "```python\n" + text + "\n```".strip()



def parse_args():
    parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
    parser.add_argument(
        "--domain",
        type=str,
        default="math",
        help="The name of the domain [math,agent,logic].",
    )
    parser.add_argument(
        "--iter_num",
        type=int,
        default=1,
        help="The number of iteration for the self-training.",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="gsm_math_full_v17",
        help="The prefix for the dataset name, directory name and save path",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7B",
        help="Model size",
    )
    parser.add_argument(
        "--cur_iter",
        type=int,
        default=10,
        help="The index of the current iteration",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    """
    load original ground-truth data
    """
    part_num = 8
    ground_truth = []
    for i in range(1, part_num+1):
        part_name = f"part{i}"
        with open(f"symbol-llm-v2/open-instruct/data/gsm_math_full_{part_name}.json", 'r') as file:
            data = json.load(file)
        ground_truth += data


    response_pool = set()
    chosen_pool = collections.defaultdict(list)
    rejected_pool = collections.defaultdict(list)
    for iter_idx in range(args.cur_iter+1):
        # [Data Load] read scores for each iteration
        scores, candidates = [], []
        scores_repaired, candidates_repaired = [], []
        for i in range(1, part_num + 1):
            if iter_idx == 0:
                part_name = f"part{i}"
                score = np.load(f"symbol-llm-v2/score_memory/{args.task_prefix}/scores_gsm_math_full_{part_name}_iter{iter_idx}.npy").tolist()
                with open(f"symbol-llm-v2/score_memory/{args.task_prefix}/gsm_math_full_{part_name}_iter{iter_idx}.json",'r') as file:
                    data = json.load(file)
                scores += score
                candidates += data
            else:
                part_name = f"part{i}"
                score = np.load(f"symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part_name}_iter{iter_idx}.npy").tolist()
                with open(f"new_generated_data/{args.task_prefix}_{part_name}_iter{iter_idx}.json", 'r') as file:
                    data = json.load(file)
                scores += score
                candidates += data

                # load self-repaired samples
                score_repaired = np.load(f"symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part_name}_iter{iter_idx}_repaired.npy").tolist()
                with open(f"new_generated_data/{args.task_prefix}_{part_name}_iter{iter_idx}_repaired.json", 'r') as file:
                    data_repaired = json.load(file)
                scores_repaired += score_repaired
                candidates_repaired += data_repaired
        print(f"Original samples: {len(scores)}")

        # merge self-repaired samples
        effective_samples = 0
        if iter_idx >= 1:
            print(f"orginal effective samples: {count_effective_samples(scores)}")
            effective_samples = count_effective_samples(scores)
            for i in range(len(scores)):
                for j in range(5):
                    if scores[i][j] != 1 and scores_repaired[i][j] == 1 or \
                            (scores[i][j]==1 and scores_repaired[i][j]==1 and candidates[i*5+j]['logprobs']<candidates_repaired[i*5+j]['logprobs']):
                        scores[i][j] = 1
                        candidates[i * 5 + j] = candidates_repaired[i * 5 + j]
            print(f"orginal effective samples after self-repair: {count_effective_samples(scores)}")



        include_id = set()
        freq_id = defaultdict(int)
        explored_id = set()
        preference_data = []
        preference_data_sft = []
        preference_data_filtered = []
        preference_data_best = []
        # [All] get chosen pool and rejected pool
        explored_id_cur = set()
        repair_id_cur = set()
        for i in range(len(scores)):  # for each origin sample, origin id: i
            chosen_candidates_idx_list = []
            rejected_candidates_idx_list = []
            for j in range(5):  # generate 5 code for each (x,y)
                response = extract_code_blocks(candidates[i*5+j]['response'])
                if scores[i][j] == 1 and response not in response_pool and (iter_idx>2 or ground_truth[i]['input'].strip() in response):
                    chosen_candidates_idx_list.append(i * 5 + j)
                    response_pool.add(response)
                elif response not in response_pool:
                    rejected_candidates_idx_list.append(i * 5 + j)
                    response_pool.add(response)

            # update chosen pool for each sample
            chosen_candidates_list, rejected_candidates_list = [], []
            for m in range(len(chosen_candidates_idx_list)):
                chosen_candidates_list.append(candidates[chosen_candidates_idx_list[m]])
            for m in range(len(rejected_candidates_idx_list)):
                rejected_candidates_list.append(candidates[rejected_candidates_idx_list[m]])

            chosen_pool[str(i)] = sorted(chosen_pool[str(i)]+chosen_candidates_list, key=lambda x: x.get('logprobs', float('-inf')), reverse=True)
            rejected_pool[str(i)] = sorted(rejected_pool[str(i)]+rejected_candidates_list, key=lambda x: x.get('logprobs', float('-inf')), reverse=True)


            if chosen_pool[str(i)]:
                # [SFT] normal sft data
                explore_num = 0
                for m in range(min(10, max(1,len(chosen_pool[str(i)])-1))):  # choose at most 10 candidates
                    data_dict = {}
                    data_dict['origin_id'] = str(i)
                    data_dict['source'] = ground_truth[i]['source']
                    data_dict['type'] = "self-explore"
                    data_dict['prompt'] = "Write a Python code to solve the problem.\n" + "\nThe question is:" + \
                                          ground_truth[i]['input'] + "\nThe solution code is:\n"
                    data_dict['completion'] = extract_code_blocks(chosen_pool[str(i)][m]['response'])
                    explore_num += 1
                    explored_id_cur.add(str(i))
                    include_id.add(str(i))
                    preference_data_sft.append(data_dict)
                    if m==0:
                        preference_data_best.append(data_dict)


                # [SFT] self-repair sft data
                repair_num = max(0, min(2, len(chosen_pool[str(i)])-explore_num, len(rejected_pool[str(i)])))
                for m in range(repair_num):
                    data_dict = {}
                    data_dict['origin_id'] = str(i)
                    data_dict['source'] = ground_truth[i]['source']
                    data_dict['type'] = "self-repair"
                    data_dict['prompt'] = "You are provided with a Python code to solve the given problem. You can either repair and refine it, or you can simply return the original one.\n" + \
                                          "\nThe question is:" + ground_truth[i]['input'] + "\nThe current Python code is:\n" + \
                                          extract_code_blocks(rejected_pool[str(i)][m]['response']) + \
                                          "\nThe solution code is:\n"
                    data_dict['completion'] = extract_code_blocks(chosen_pool[str(i)][explore_num+m]['response'])
                    repair_id_cur.add(str(i))
                    preference_data_sft.append(data_dict)


        print(f"The iteration {iter_idx} has: DPO data {len(preference_data_filtered)}, SFT data {len(preference_data_sft)}")
        print(f"The current normal sft data: {len(explored_id_cur)}, self-repair sft data: {len(repair_id_cur)}")
        print(f"The number of used samples: {len(include_id)}")
        #     print(f"Current iteration contains: {len(include_id_gsm)} gsm samples, and {len(include_id_math)} math samples, {len(include_id_gsm)/len(include_id_math)}")
        print("-" * 30)


    with open(f"symbol-llm-v2/logs/{args.task_prefix}_log.txt","a") as file:
        file.write(f"orginal effective samples before self-repair: {effective_samples}\n")
        file.write(f"orginal effective samples after self-repair: {count_effective_samples(scores)}\n")
        file.write(f"The iteration {iter_idx} has: SFT data {len(preference_data_sft)}\n")
        file.write(f"The current normal sft data: {len(explored_id_cur)}, self-repair sft data: {len(repair_id_cur)}\n")
        file.write(f"The number of used samples: {len(include_id)}\n")
        file.write("-" * 30)
        file.write("\n")


    with jsonlines.open(f"symbol-llm-v2/open-instruct/data/{args.task_prefix}_sft_iter{args.cur_iter}.jsonl",'w') as file:
        random.shuffle(preference_data_sft)
        for i in range(len(preference_data_sft)):
            file.write(preference_data_sft[i])


if __name__ == "__main__":
    main()
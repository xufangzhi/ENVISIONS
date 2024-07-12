import os

os.chdir(f"symbol-llm-v2/pal/")

import json
import re
import pal
import numpy as np
import time
import signal

import sys
import argparse
import logging

logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)



def parse_args():
    parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
    parser.add_argument(
        "--domain",
        type=str,
        default="math",
        help="The name of the domain [math,agent,logic].",
    )
    parser.add_argument(
        "--cur_iter",
        type=int,
        default=0,
        help="The index of the current iteration",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="gsm_math_full_v10",
        help="The prefix for the dataset name, directory name and save path",
    )
    parser.add_argument(
        "--repaired",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.repaired:
        suffix = "_repaired"
    else:
        suffix = ""

    interface = pal.interface.ProgramInterface(
        model='gpt-3.5-turbo-0301',
        stop='\n\n\n', # stop generation str for Codex API
        get_answer_expr='solution()', # python expression evaluated after generated code to obtain answer
        # get_answer_expr='',
        # get_answer_from_stdout=True,
    )

    for part_index in range(1,6):
        part = f"part{part_index}"
        # with open(f"../Preference_Dataset/metamathqa_add_{part}_symbolllm_instruct.json") as file:
        #     pred = json.load(file)
        # with open(f"../SymbolLLMv2/open-instruct/data/new_metamathqa_preference_data_{part}.json") as file:
        #     gold = json.load(file)
        # with open(f"../Preference_Dataset/theoremqa/theoremqa_v2_iter1_repaired.json") as file:
        with open(f"../score_memory/gsm_math_full/{args.task_prefix}_{part}_iter{args.cur_iter+1}{suffix}.json") as file:
        # with open(f"../Preference_Dataset/gsm_math_full/gsm_math_full_symbolllm_instruct_{part}_iter1.json") as file:
            pred = json.load(file)
        # with open(f"../SymbolLLMv2/open-instruct/data/theoremqa_train.json") as file:
        with open(f"../open-instruct/data/gsm_math_full_{part}.json") as file:
            gold = json.load(file)
        # gold = gold[:10]
        # with open(f"../T2S_Dataset/{dataset}/{subdataset}_test.json") as file:
        #     gold = json.load(file)
        print(len(pred),len(gold))
        assert len(pred)//5==len(gold), "The length between pred and gold is not equal"

        num = 0
        preference_scores_list = []

        # for each sample
        for i in range(len(gold)):
            preference_scores_each_sample = []
            for j in range(5):
                code = pred[i*5+j]['response']
                ground_truth = gold[i]['label']
                print(i,j)
                try:
                    def extract_code_blocks(text):
                        pattern = r"```(.*?)```"
                        code_blocks = re.findall(pattern, text, re.DOTALL)
                        if code_blocks:
                            if code_blocks[0].startswith("python\n"):
                                return code_blocks[0].split("python\n")[1]
                            else:
                                return code_blocks[0]
                        else:
                            return text

                    def handler(signum, frame):
                        raise TimeoutError("Function execution timed out.")

                    def run_with_timeout(func, timeout):
                        # 设置信号处理程序
                        signal.signal(signal.SIGALRM, handler)
                        signal.alarm(timeout)  # 设置闹钟
                        start_time = time.time()
                        try:
                            result = func()  # 运行黑盒函数
                        except TimeoutError:
                            print("Function execution timed out.")
                            # signal.alarm(0)  # 取消闹钟
                            result = None
                        finally:
                            signal.alarm(0)  # 取消闹钟
                        return result


                    code = extract_code_blocks(code)
                    def func():
                        if "itertools" not in code and "input(" not in code and "plt." not in code:
                            return interface.execute(["def solution()"+code.split("def solution()")[1].strip()])
                        else:
                            return None
                    # answer = func()
                    answer = run_with_timeout(func, 2)  # signal.alarm 只能接受整数
                    # print(d['response'])
                    print(f"---- {i/len(gold)} ----")
                    print(answer)
                    if answer and abs(float(answer)-float(ground_truth))<=0.0001 or answer==ground_truth:
                        num += 1
                        preference_scores_each_sample.append(1)
                    else:
                        if answer:
                            preference_scores_each_sample.append(0)
                        else:
                            preference_scores_each_sample.append(2)   # execution time exceed
                except:
                    print("fail to excute")
                    preference_scores_each_sample.append(-1)    # program fails, score -1

            preference_scores_list.append(preference_scores_each_sample)

        # np.save(f"../Preference_Dataset/theoremqa/scores_theoremqa_v2_iter1_repaired.npy",np.array(preference_scores_list))
        # np.save(f"../Preference_Dataset/gsm_math_full/scores_gsm_math_full_symbolllm_instruct_{part}_iter1.npy",np.array(preference_scores_list))
        np.save(f"../score_memory/gsm_math_full/scores_{args.task_prefix}_{part}_iter{args.cur_iter+1}{suffix}.npy",np.array(preference_scores_list))


        # print(num/len(gold)/5)
        # print(len(gold))

if __name__ == "__main__":
    main()
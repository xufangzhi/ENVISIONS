import json
import re
import pal
import time
import signal
import jsonlines

interface = pal.interface.ProgramInterface(
    model='gpt-3.5-turbo-0301',
    stop='\n\n\n', # stop generation str for Codex API
    get_answer_expr='solution()', # python expression evaluated after generated code to obtain answer
    # get_answer_expr='',
    # get_answer_from_stdout=True,
)

data = []
with open(f"../api_call/results/gpt4-turbo/gsm_math_full_gpt-4-0125-preview.jsonl") as file:
# with open(f"../api_call/results/gpt-4-0125-preview/theoremqa_gpt-4-0125-preview.jsonl") as file:
    for line in file:
        data.append(json.loads(line))

with open(f"../Preference_Dataset/gsm_math_full/gsm_math_full_train.json") as file:
# with open(f"../Preference_Dataset/theoremqa/theoremqa_train.json") as file:
    gold = json.load(file)
# with open(f"../T2S_Dataset/{dataset}/{subdataset}_test.json") as file:
#     gold = json.load(file)
print(len(gold))
num = 0
math_num = 0
data_correct = []
for i, (d,g) in enumerate(zip(data,gold)):
    # print(d)
    try:
        code = d['response']
        ground_truth = g['label']
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
                return None
            finally:
                signal.alarm(0)  # 取消闹钟
            # print(result)
            return result

        code = extract_code_blocks(code)
        def func():
            if "itertools" not in code:
                return interface.execute(["def solution()" + code.split("def solution()")[1].strip()])
            else:
                return None


        answer = run_with_timeout(func, 2)  # signal.alarm 只能接受整数
        # print(d['response'])
        print(f"---- {i/len(data)} ----")
        print(answer)
        # print(g['target'])
        # print(answer)
        # print(g['target_scores'])
        # for ans in answer.split():
        #     if ans in g['target_scores'] and g['target_scores'][ans]==1:
        #         num += 1
        #         break
        def check_lists_equal(list1, list2):
            if isinstance(list1, list) and isinstance(list2, list):
                if len(list1) != len(list2):
                    return False
                for item1, item2 in zip(list1, list2):
                    if not check_lists_equal(item1, item2):
                        return False
            elif abs(float(list1)-float(list2))>0.0001:
                    return False
            else:
                return False
            return True

        if type(ground_truth)==list and check_lists_equal(ground_truth,answer):
            num += 1
            d['prompt'] = "Write Python code to solve the question.The following are three examples for reference.\n" \
                          + "The question: " + d['prompt'].split("The question:")[-1] + "\nThe solution code is:\n"
            d['completion'] = code
            del d['response']
            data_correct.append(d)
            # if g['source'] == "MATH":
            #     math_num += 1
        elif abs(float(answer)-float(ground_truth))<=0.0001 or answer==ground_truth:
            num += 1
            d['prompt'] = "Write Python code to solve the question.The following are three examples for reference.\n" \
                          + "The question: " + d['prompt'].split("The question:")[-1] + "\nThe solution code is:\n"
            d['completion'] = code
            del d['response']
            data_correct.append(d)

    except:
        print(111)

print("===")
print(num/len(data), math_num/len(data))
print(len(gold))

# with jsonlines.open(f"../Preference_Dataset/gsm_math_full/gsm_math_full_gpt-3.5-turbo-0125.jsonl",'w') as file:
# with jsonlines.open(f"../Preference_Dataset/theoremqa/theoremqa_gpt-4-0125-preview.jsonl", 'w') as file:
#     for d in data_correct:
#         file.write(d)
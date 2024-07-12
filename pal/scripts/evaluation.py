import json
import re
import pal
import time
import signal

dataset = "asdiv"
subdataset = "multiarith"
interface = pal.interface.ProgramInterface(
    model='gpt-3.5-turbo-0301',
    stop='\n\n\n', # stop generation str for Codex API
    get_answer_expr='solution()', # python expression evaluated after generated code to obtain answer
    # get_answer_expr='',
    # get_answer_from_stdout=True,
)

with open(f"../T2S_Dataset/{dataset}/result.json") as file:
    data = json.load(file)
with open(f"../T2S_Dataset/{dataset}/{dataset}_test.json") as file:
    gold = json.load(file)
# with open(f"../T2S_Dataset/{dataset}/{subdataset}_test.json") as file:
#     gold = json.load(file)

verified_list = []
num = 0
for i, (d,g) in enumerate(zip(data,gold)):
    # print(d)
    data_dict = {}
    data_dict['input'] = g['input']
    data_dict['response'] = ""
    data_dict['correctness'] = False
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
                return text.strip()


        code = extract_code_blocks(d['response'])
        ground_truth = g['target'] if "target" in g else g['label']
        data_dict['response'] = "```python\n"+code+"\n```".strip()


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

        def func():
            if "itertools" not in code:
                return interface.execute(["def solution()" + code.split("def solution()")[1].strip()])
            else:
                return None


        # answer = run_with_timeout(func, 5)  # signal.alarm 只能接受整数
        # print(d['response'].split("```")[1].strip())
        # answer = interface.execute([d['response'].split("```")[1]])
        # print(extract_code_blocks(d['response'].strip()))
        # print(d['response'].strip("\n\nsolution()"))

        # answer = interface.execute([d['response'].strip()])


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
            data_dict['correctness'] = True
        elif abs(float(answer)-float(ground_truth))<=0.0001 or answer==ground_truth:
            num += 1
            data_dict['correctness'] = True
        # if answer.strip()==g['answer']:   # last
        #     num += 1
        # if float(answer)==float(g['target']):
        #     num += 1
    except:
        print(111)
    verified_list.append(data_dict)
print(num/len(gold))
print(len(gold))


print(len(verified_list))

# with open(f"../SymbolLLMv2/verified_results/{dataset}/gsm_math_full_v10_iter9.json",'w') as file:
#     json.dump(verified_list,file,indent=4)
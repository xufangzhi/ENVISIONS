import json
import re
import pal
import time
import signal


interface = pal.interface.ProgramInterface(
    model='gpt-3.5-turbo-0301',
    stop='\n\n\n', # stop generation str for Codex API
    # get_answer_expr='solution()', # python expression evaluated after generated code to obtain answer
    get_answer_expr='',
    get_answer_from_stdout=True,
)

with open(f"../Preference_Dataset/new_math.json",'r') as file:
    data = json.load(file)

num = 0
preference_data = []
for d in data:
    data_dict = {}
    try:
        answer = interface.execute([d['output'].strip()])
        print("----")
        print(answer)
        data_dict['dataset'] = d['source']
        data_dict['id'] = str(num)
        data_dict['input'] = d['instruction'] + " The solution code is:\n"
        data_dict['label'] = answer
        preference_data.append(data_dict)
        num += 1
    except:
        print(111)

print(num)

# build preference dataset with format

for j in range(num//5000):
    with open(f"../SymbolLLMv2/open-instruct/data/new_math_preference_data_part{j+1}.json",'w') as file:
        json.dump(preference_data[j*5000:(j+1)*5000], file, indent=4)

with open(f"../SymbolLLMv2/open-instruct/data/new_math_preference_data_part{j+1}.json",'w') as file:
    json.dump(preference_data[j*5000:], file, indent=4)

len(preference_data)

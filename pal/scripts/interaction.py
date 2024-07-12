import pal
import time
import signal

interface = pal.interface.ProgramInterface(
    model='gpt-3.5-turbo-0301',
    stop='\n\n\n', # stop generation str for Codex API
    get_answer_expr='solution()', # python expression evaluated after generated code to obtain answer
    # get_answer_expr=""
    get_answer_from_stdout=True
)



# question = 'add 10 and 1.'
# prompt = math_prompts.MATH_PROMPT.format(question=question)

code = ["a=1\nprint(a)"]
answer = interface.execute(code)
print(answer)
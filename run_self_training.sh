source /cpfs01/user/xufangzhi/anaconda3/bin/activate /cpfs01/user/xufangzhi/anaconda3/envs/vllm
#cp /cpfs01/user/xufangzhi/symbol-llm-v2/chromedriver-linux64/chromedriver -r /usr/bin
#echo 'export PATH=$PATH:/cpfs01/user/xufangzhi/symbol-llm-v2/chromedriver-linux64/chromedriver' >> ~/.bash_profile
#source ~/.bash_profile
#chromedriver --version
echo "[INFO] We have successfully activate the environment."
echo "[INFO] Start to run the shell."

python symbol-llm-v2/self_training_miniwob_star.py --base_model "llama2chat" --model_size "13B" --task_prefix "miniwob_star_llama2chat13b" --vllm_batchsize 1
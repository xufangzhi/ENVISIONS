echo "[INFO] We have successfully activate the environment."
echo "[INFO] Start to run the shell."

python ENVISIONS/self_training_miniwob.py --base_model "llama2chat" --model_size "7B" --task_prefix "gsm_math_full_llama2chat" --vllm_batchsize 1
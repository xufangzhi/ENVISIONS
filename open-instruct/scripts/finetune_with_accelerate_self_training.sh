source /cpfs01/user/xufangzhi/anaconda3/bin/activate /cpfs01/user/xufangzhi/anaconda3/envs/flashattv2
cd symbol-llm-v2/open-instruct
echo "[INFO] We have successfully activate the environment."
echo "[INFO] Start to run the shell."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TASK_PREFIX=$1
ITER_NUM=$2
BASE_MODEL=$3


#MODEL_DIR=/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct
#MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v8_dpo_iter4_dpo_tune_sft_iter3_7B
#MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/theoremqa_sft_iter1_sft_tune_sft_iter0_7B
MODEL_SIZE=$4
DS_FILE=ds_configs/stage3_no_offloading_accelerate.conf
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

echo "$BASE_MODEL"
echo "$ITER_NUM"

if [ "$BASE_MODEL" = "llama2chat" ] || [ "$ITER_NUM" = "0" ]; then
  if [ "$MODEL_SIZE" = "7B" ]; then
    MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf
  elif [ "$MODEL_SIZE" = "13B" ]; then
    MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf
    DS_FILE=ds_configs/stage3_offloading_accelerate.conf
  fi
elif [ "$BASE_MODEL" = "cont" ]; then
  PREV_ITER_NUM=$((ITER_NUM - 1))
  MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/${TASK_PREFIX}_sft_iter${PREV_ITER_NUM}_sft_tune_${BASE_MODEL}_${MODEL_SIZE}
fi


if [ "$BASE_MODEL" = "llemma" ]; then
  MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--EleutherAI--llemma_7b/snapshots/e223eee41c53449e6ea6548c9b71c50865e4a85c
  MODEL_SIZE=7B
fi

if [ "$BASE_MODEL" = "symbolllm" ]; then
  MODEL_DIR=/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct
  MODEL_SIZE=7B
fi


if [ "$BASE_MODEL" = "mammoth" ]; then
  MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--TIGER-Lab--MAmmoTH-Coder-7B/snapshots/857751d2f6255ec1bf5e76bac24cbc24a67288ae
  MODEL_SIZE=7B
fi


if [ "$BASE_MODEL" = "well" ]; then
  MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_gpt-3.5-turbo-0125_7B
  MODEL_SIZE=7B
fi


if [ "$BASE_MODEL" = "vicuna" ]; then
  MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5
  MODEL_SIZE=7B
fi


if [ "$BASE_MODEL" = "gpt2xl" ]; then
  MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8
  MODEL_SIZE=1.5B
fi


if [ "$BASE_MODEL" = "tinyllama" ]; then
  MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6
  MODEL_SIZE=1.1B
fi

if [ "$BASE_MODEL" = "opt" ]; then
  if [ "$MODEL_SIZE" = "2.7B" ]; then
    MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--facebook--opt-2.7b/snapshots/905a4b602cda5c501f1b3a2650a4152680238254
    MODEL_SIZE=2.7B
  elif [ "$MODEL_SIZE" = "1.3B" ]; then
    MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62
    MODEL_SIZE=1.3B
  fi
fi


if [ "$BASE_MODEL" = "deepseekchat" ]; then
  if [ "$MODEL_SIZE" = "7B" ]; then
    MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--deepseek-ai--deepseek-llm-7b-chat/snapshots/afbda8b347ec881666061fa67447046fc5164ec8
    MODEL_SIZE=7B
  fi
fi

if [ "$BASE_MODEL" = "llama3chat" ]; then
  if [ "$MODEL_SIZE" = "8B" ]; then
    MODEL_DIR=/cpfs01/shared/NLP-A100/NLP-A100_hdd/model/Meta-Llama-3-8B
    MODEL_SIZE=8B
  fi
fi

if [ "$BASE_MODEL" = "codellama34b" ]; then
  if [ "$MODEL_SIZE" = "34B" ]; then
    MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d650b03778bb6cce1d21805bb757e4d42d222574
    MODEL_SIZE=34B
    DS_FILE=ds_configs/stage3_offloading_accelerate.conf
  fi
fi

echo "$MODEL_DIR"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ${DS_FILE} \
    open_instruct/finetune.py \
    --model_name_or_path ${MODEL_DIR} \
    --use_flash_attn \
    --tokenizer_name ${MODEL_DIR} \
    --use_slow_tokenizer \
    --train_file ./data/${TASK_PREFIX}_sft_iter${ITER_NUM}.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ./output/${TASK_PREFIX}_sft_iter${ITER_NUM}_sft_tune_${BASE_MODEL}_${MODEL_SIZE} \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
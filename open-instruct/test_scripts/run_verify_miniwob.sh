source /cpfs01/user/xufangzhi/anaconda3/bin/activate /cpfs01/user/xufangzhi/anaconda3/envs/vllm

echo "[INFO] We have successfully activate the environment."
echo "[INFO] Start to run the shell."

PART_ID=$1
REPAIRED=$2

if [ "$REPAIRED" = "TRUE" ]; then
  python symbol-llm-v2/Synapse/label_preference.py --part_id ${PART_ID} --repaired
else
  python symbol-llm-v2/Synapse/label_preference.py --part_id ${PART_ID}
fi
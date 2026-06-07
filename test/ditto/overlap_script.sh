MODEL_PATH=/models/Llama-3-8B-Instruct \
METHOD=offloading \
DITTO_RECORD_OVERLAP_STATS=1 \
DITTO_TRANSFER_STATS_FILE=/tmp/llama_ditto_overlap_stats.json \
CUDA_VISIBLE_DEVICES=7 \
bash test_accuracy.sh

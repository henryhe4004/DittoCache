MODEL_PATH=/jhe/Llama-3-8B-Instruct-Gradient-1048k \
METHOD=offloading \
LITECACHE_RECORD_OVERLAP_STATS=1 \
LITECACHE_TRANSFER_STATS_FILE=/tmp/llama_litecache_overlap_stats.json \
CUDA_VISIBLE_DEVICES=7 \
bash test_accuracy.sh

run_dirs=$1
task_name=$2
sub_server_num=$3
port=$4
device=$5

if [ ${task_name} == "qnli" ];
then
  max_seq=256
else
  max_seq=128
fi

echo "${task_name}'s max_seq is ${max_seq}"

let "word_size=${sub_server_num}+1"

echo "word_size is ${word_size}"

CUDA_VISIBLE_DEVICES=${device} python main.py \
--model_name_or_path ${run_dirs}/pretrain/nlp/bert-base-uncased/ \
--output_dir ${run_dirs}/output/fedglue \
--rank 0 \
--task_name ${task_name} \
--fl_algorithm fedavg \
--model_type bert \
--raw_dataset_path ${run_dirs}/data/fedglue \
--partition_dataset_path ${run_dirs}/data/fedglue \
--per_device_train_batch_size 32 \
--dataset_name glue \
--model_output_mode seq_classification \
--num_train_epochs 1 \
--max_seq_length ${max_seq} \
--world_size ${word_size} \
--port ${port} &

sleep 2s

for ((i=1; i<=${sub_server_num}; i++))
do
{
    echo "client ${i} started"
    CUDA_VISIBLE_DEVICES=${device} python main.py \
    --model_name_or_path ${run_dirs}/pretrain/nlp/bert-base-uncased/ \
    --output_dir ${run_dirs}/output/fedglue \
    --rank ${i} \
    --task_name ${task_name} \
    --fl_algorithm fedavg \
    --model_type bert \
    --raw_dataset_path ${run_dirs}/data/fedglue \
    --partition_dataset_path ${run_dirs}/data/fedglue \
    --per_device_train_batch_size 32 \
    --dataset_name glue \
    --model_output_mode seq_classification \
    --num_train_epochs 1 \
    --max_seq_length ${max_seq} \
    --world_size ${word_size} \
    --port ${port} &
    sleep 2s
}
done

wait
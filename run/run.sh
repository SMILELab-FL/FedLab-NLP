run_dirs=$1
task_name=$2
port=$3
device=$4

CUDA_VISIBLE_DEVICES=${device} python main.py \
--model_name_or_path ${run_dirs}/pretrain/nlp/bert-base-uncased/ \
--output_dir ${run_dirs}/output/fedmask \
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
--world_size 3 \
--port ${port} &

sleep 2s

CUDA_VISIBLE_DEVICES=${device} python main.py \
--model_name_or_path ${run_dirs}/pretrain/nlp/bert-base-uncased/ \
--output_dir ${run_dirs}/output/fedmask \
--rank 1 \
--task_name ${task_name} \
--fl_algorithm fedavg \
--model_type bert \
--raw_dataset_path ${run_dirs}/data/fedglue \
--partition_dataset_path ${run_dirs}/data/fedglue \
--per_device_train_batch_size 32 \
--dataset_name glue \
--model_output_mode seq_classification \
--num_train_epochs 1 \
--world_size 3 \
--port ${port} &

CUDA_VISIBLE_DEVICES=${device} python main.py \
--model_name_or_path ${run_dirs}/pretrain/nlp/bert-base-uncased/ \
--output_dir ${run_dirs}/output/fedmask \
--rank 2 \
--task_name ${task_name} \
--fl_algorithm fedavg \
--model_type bert \
--raw_dataset_path ${run_dirs}/data/fedglue \
--partition_dataset_path ${run_dirs}/data/fedglue \
--per_device_train_batch_size 32 \
--dataset_name glue \
--model_output_mode seq_classification \
--num_train_epochs 1 \
--world_size 3 \
--port ${port}

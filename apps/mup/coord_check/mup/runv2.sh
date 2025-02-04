#!/bin/bash

# Function to run a single training job
run_training() {
    local width=$1
    local seed=$2
    local gpu_id=$3

    head_size=64
    n_heads=$((width / head_size))
    mup_base_width=256
    mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
    out_dir="apps/mup/coord_check/mup/out/width${width}_depth2_seed${seed}"
    # randomly select a master port
    master_port=$(( ( RANDOM % 1000 )  + 28000 ))

    CUDA_VISIBLE_DEVICES=$gpu_id python -m apps.mup.train \
        name=mup \
        use_mup=true \
        dump_dir=$out_dir \
        steps=10 \
        probe_freq=null \
        seed=$seed \
        optim.lr=3e-3 \
        optim.weight_decay=0.1 \
        optim.clip=1.0 \
        optim.warmup=0 \
        distributed.fsdp_type=no_shard \
        distributed.compile=true \
        distributed.model_dtype=bf16 \
        distributed.matmul_allow_tf32=false \
        distributed.selective_activation_checkpointing=false \
        distributed.tp_size=1 \
        distributed.master_port=$master_port \
        model.dim=$width \
        model.n_layers=2 \
        model.n_heads=$n_heads \
        model.mup_dim_model_base=256 \
        data.root_dir=/mnt/vast/pretraining-data-jsonl/ \
        "data.sources={english/dclm_crossdeduped/shard_000: 100.0}" \
        data.batch_size=4 \
        data.prefetch_size=1024 \
        data.seq_len=4096 \
        data.n_views=2 \
        data.load_async=true \
        data.add_bos=true \
        data.add_eos=true \
        data.tokenizer.name=tiktoken \
        data.tokenizer.path=tokenizers/llama3/tokenizer.model \
        checkpoint.dump.every=10000 \
        checkpoint.dump.keep=0 \
        checkpoint.eval.every=10000 \
        checkpoint.eval.keep=0 \
        logging.freq=1 &
}

# Configuration arrays
widths=(256 512 1024 2048 4096)
# seeds=(1 2 3 4 5)
seeds=(1)

# Initialize GPU counter
gpu_id=0
job_count=0

# Run all combinations
for width in "${widths[@]}"; do
    for seed in "${seeds[@]}"; do
        # Run the training job on the current GPU
        run_training $width $seed $gpu_id
        echo "Started job: width=$width, seed=$seed on GPU $gpu_id"
        
        # Increment GPU counter and wrap around if necessary
        gpu_id=$((gpu_id + 1))
        if [ $gpu_id -eq 8 ]; then
            gpu_id=0
        fi
        
        # Increment job counter
        job_count=$((job_count + 1))
        
        # If we've started 8 jobs, wait for all to complete before continuing
        if [ $((job_count % 8)) -eq 0 ]; then
            wait
            echo "Batch of 8 jobs completed. Starting next batch..."
        fi
    done
done

# Wait for any remaining jobs to complete
wait
echo "All training jobs completed!"
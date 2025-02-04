for width in 256 512 1024 2048 4096; do
# for width in 256; do
    for seed in 1 2; do
    # for seed in 1 2 3 4 5; do
    # for seed in 1; do
        head_size=64
        n_heads=$((width / head_size))
        mup_base_width=256
        mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
        out_dir="apps/mup/coord_check/sp/out/width${width}_depth2_seed${seed}"

        python -m apps.mup.train \
            name=sp \
            use_mup=false \
            dump_dir=$out_dir \
            steps=10 \
            probe_freq=null \
            seed=$seed \
            optim.lr=3e-3 \
            optim.weight_decay=0.1 \
            optim.clip=1.0 \
            distributed.fsdp_type=no_shard \
            distributed.compile=true \
            distributed.model_dtype=bf16 \
            distributed.matmul_allow_tf32=false \
            distributed.selective_activation_checkpointing=false \
            distributed.tp_size=1 \
            model.dim=$width \
            model.n_layers=2 \
            model.n_heads=$n_heads \
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
            logging.freq=1
        done
    done
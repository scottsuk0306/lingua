import os
import random
import subprocess
import time
from itertools import product
from typing import List

from apps.mup.train import TrainArgs
from omegaconf import DictConfig, OmegaConf


def get_default_config() -> DictConfig:
    config = OmegaConf.structured(TrainArgs())

    # Set specific defaults for MuP experiments
    config.name = "sp"
    config.steps = 10
    config.dump_dir = "apps/mup/out"

    # Model settings
    config.model.dim = 256
    config.model.n_layers = 2
    config.model.n_heads = 8

    # Optimizer settings
    config.optim.lr = 3e-3
    config.optim.weight_decay = 0.1
    config.optim.clip = 1.0
    config.optim.warmup = 0

    # Distributed settings
    config.distributed.master_port = random.randint(28000, 29000)

    # Data settings
    config.data.root_dir = "/mnt/vast/pretraining-data-jsonl/"
    config.data.sources = {"english/dclm_crossdeduped/shard_000": 100.0}
    config.data.batch_size = 4
    config.data.prefetch_size = 1024
    config.data.seq_len = 4096
    config.data.n_views = 2
    config.data.load_async = True
    config.data.add_bos = True
    config.data.add_eos = True
    config.data.tokenizer.name = "tiktoken"
    config.data.tokenizer.path = "tokenizers/llama3/tokenizer.model"

    # Checkpoint settings
    config.checkpoint.dump.every = 10000
    config.checkpoint.dump.keep = 0
    config.checkpoint.eval.every = 10000
    config.checkpoint.eval.keep = 0

    # Logging settings
    config.logging.freq = 1

    return config


def get_command(config: DictConfig) -> List[str]:
    """Convert OmegaConf config to command line arguments."""
    cmd = ["python", "-m", "apps.mup.train"]

    # Convert config to flat dictionary with dot notation.
    flat_config = OmegaConf.to_container(config, resolve=True)

    def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Flatten the nested dictionary.
    flat_dict = flatten_dict(flat_config)

    # Add all parameters as command-line arguments.
    for key, value in flat_dict.items():
        if value is not None:  # Skip None values
            cmd.append(f"{key}={value}")

    return cmd


def run_training(config: DictConfig, gpu_id: int):
    """Run a training job using the given config on the given GPU."""
    cmd = get_command(config)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # You can uncomment this line to actually run the job.
    # return subprocess.Popen(cmd, env=env)
    
    # For demonstration, we'll just print the command here:
    print(f"\n\nRunning on GPU {gpu_id}: {' '.join(cmd)}\n\n")
    return None  # Return None since we're not actually running.


def main():
    n_gpus = 8
    max_concurrent_jobs = n_gpus

    search_space = {
        "model.dim": [256, 512],  # e.g. widths
        "model.n_layers": [2, 4],
        "optim.lr": [3e-3, 1e-3],
    }

    configs = []

    use_mup_options = [True, False]

    keys = list(search_space.keys())  # e.g. ["model.dim", "model.n_layers", "optim.lr"]
    values_lists = [search_space[k] for k in keys]  # e.g. [[256, 512], [2, 4], [3e-3, 1e-3]]

    for combination in product(*values_lists):
        for use_mup in use_mup_options:
            # Create a fresh config for each combination.
            cfg = get_default_config()

            # We need to set each hyperparameter in the config.
            for key, value in zip(keys, combination):
                subkeys = key.split('.')

                current = cfg
                for sk in subkeys[:-1]:  # walk down to the penultimate key
                    current = current[sk]

                current[subkeys[-1]] = value

            # Also set use_mup
            cfg.use_mup = use_mup

            # Make sure the names are unique
            # We'll build a string that includes all hyperparameter info
            # plus whether it's mup or sp.
            # Example: "mup_dim256_layers2_lr0.003"
            name_parts = ["mup" if use_mup else "sp"]
            for key, val in zip(keys, combination):
                safe_key = key.replace('.', '_')
                name_parts.append(f"{safe_key}{val}")

            cfg.name = "_".join(name_parts)
            configs.append(cfg)

    # Print out how many configs we have:
    print(f"Generated {len(configs)} configurations.")

    # Run them with concurrency control.
    active_processes = []
    job_count = 0

    for config in configs:
        # If we have max jobs running, wait for one to complete.
        while len(active_processes) >= max_concurrent_jobs:
            # Check each process and remove completed ones
            still_active = []
            for p in active_processes:
                if p.poll() is None:  # None => still running
                    still_active.append(p)
            active_processes = still_active

            # Sleep a bit before re-checking
            if len(active_processes) >= max_concurrent_jobs:
                time.sleep(1)

        # Start new job
        gpu_id = job_count % n_gpus
        process = run_training(config, gpu_id)
        if process is not None:
            active_processes.append(process)

        print(f"Started job {config.name} on GPU {gpu_id}")
        job_count += 1

    # Wait for remaining jobs to complete
    print("Waiting for remaining jobs to complete...")
    for process in active_processes:
        process.wait()

    print("All training jobs completed!")


# python -m apps.mup.coord_check_shape.run
if __name__ == "__main__":
    main()

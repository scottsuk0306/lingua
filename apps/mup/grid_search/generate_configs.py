import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class TrainingConfig:
    """Configuration for a single training run"""
    width: int
    lr: float
    seed: int
    scale_depth: float
    scale_emb: float
    init_std: float
    base_width: int = 256
    head_size: int = 64

    @property
    def n_heads(self) -> int:
        return self.width // self.head_size

    @property
    def mup_width_multiplier(self) -> float:
        return self.width / self.base_width

def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load the base configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_config_for_run(base_config: Dict[str, Any], config: TrainingConfig) -> Dict[str, Any]:
    """Create a new config dictionary with modified parameters."""
    new_config = base_config.copy()
    
    # Update model configuration
    if 'model' not in new_config:
        new_config['model'] = {}
    
    new_config['model'].update({
        'mup_scale_emb': config.scale_emb,
        'mup_scale_depth': config.scale_depth,
        'init_base_std': config.init_std,
        'mup_dim_model_base': config.base_width,
        'width': config.width,
        'n_heads': config.n_heads,
    })
    
    # Update optimizer configuration
    if 'optim' not in new_config:
        new_config['optim'] = {}
    new_config['optim']['lr'] = config.lr
    
    # Update training configuration
    new_config['seed'] = config.seed
    new_config['use_mup'] = True
    
    # Generate unique name for this configuration
    new_config['name'] = f"mup_width{config.width}_depth{config.scale_depth}_seed{config.seed}_lr{config.lr}"
    
    return new_config

def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save the configuration to a YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def main():
    # Parameter lists for grid search
    scale_depth_list = [
        1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7,
        4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8, 6.1, 6.4, 6.7,
        7.0, 7.3, 7.6, 7.8, 8.0
    ]

    scale_emb_list = [
        2.5, 3.2, 3.9, 4.6, 5.3, 6.0, 6.7, 7.4, 8.1, 8.8,
        9.5, 10.2, 10.9, 11.6, 12.3, 13.0, 13.7, 14.4, 15.1, 15.8,
        16.5, 17.2, 17.9, 18.6, 20.0
    ]

    init_std_list = [
        0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009, 0.012, 0.015, 0.020,
        0.025, 0.030, 0.040, 0.050, 0.060, 0.075, 0.090, 0.105, 0.120, 0.135,
        0.150, 0.165, 0.180, 0.190, 0.200
    ]
    
    learning_rates = [
        0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009, 0.012, 0.015, 0.020,
        0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.070, 0.075, 0.080,
        0.085, 0.090, 0.095, 0.098, 0.100
    ]

    # Configuration for this run
    widths = [256]
    seeds = [1]

    # Load base configuration
    base_config_path = "apps/mup/grid_search/base_config.yaml"
    base_config = load_base_config(base_config_path)

    # Generate all combinations
    configs = [
        TrainingConfig(
            width=width,
            lr=lr,
            seed=seed,
            scale_depth=scale_depth,
            scale_emb=scale_emb,
            init_std=init_std
        )
        for width in widths
        for lr in learning_rates[:5]
        for seed in seeds
        for scale_depth in scale_depth_list[:5]  # Using first 5 for example
        for scale_emb in scale_emb_list[:5]      # Using first 5 for example
        for init_std in init_std_list[:5]        # Using first 5 for example
    ]

    # Create output directory
    output_dir = Path("apps/mup/grid_search/configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save configurations
    for i, config in enumerate(configs):
        # Create new config
        new_config = create_config_for_run(base_config, config)
        
        # Generate filename
        filename = f"config_width{config.width}_depth{config.scale_depth}_seed{config.seed}_lr{config.lr}.yaml"
        output_path = output_dir / filename
        
        # Save config
        save_config(new_config, output_path)
        
        logging.info(f"Generated config {i+1}/{len(configs)}: {filename}")

    logging.info(f"Successfully generated {len(configs)} config files in {output_dir}")

if __name__ == "__main__":
    main()
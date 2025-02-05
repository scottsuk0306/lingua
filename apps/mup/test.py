import torch.nn as nn
from apps.mup.transformer import BaseTransformer, BaseTransformerArgs
from lingua.tokenizer import build_tokenizer
from apps.mup.transformer import LMTransformer, LMTransformerArgs


def validate_model_args(dim: int, n_layers: int, n_heads: int):    
    assert n_heads * 64 == dim, "Number of heads must be dim * 64"
    
    
def load_model(dim: int, n_layers: int, n_heads: int):
    tokenizer = build_tokenizer("tiktoken", path="tokenizers/llama3/tokenizer.model")

    model_args = LMTransformerArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=tokenizer.n_words,
        weight_tying=True,
    )
    
    model = LMTransformer(model_args)
    return model


def get_num_params(model: nn.Module, non_embedding: bool = True) -> int:
    """
    Get the number of embedding and non-embedding parameters in the model
    """
    n_params = sum(p.numel() for p in model.parameters())
    
    if non_embedding:
        n_params -= model.tok_embeddings.weight.numel()
    
    # weight_tying parameters are not counted as separate parameters
    # if hasattr(model.output, "tied_module"):
    #     n_params += model.tok_embeddings.weight.numel()
    
    return n_params


def test_model_args():
    model_args = LMTransformerArgs(weight_tying=True)
    


def test_tokenizer():
    tokenizer = build_tokenizer("tiktoken", path="tokenizers/llama3/tokenizer.model")
    print(tokenizer)


def test_load():
    model = load_model(2024, 2, 8)
    
    print(model)
    
    total_params = get_num_params(model, non_embedding=False)
    non_emb_params = get_num_params(model, non_embedding=True)
    emb_params = total_params - non_emb_params

    print(f"Embedding parameters: {emb_params}")
    print(f"Non-embedding parameters: {non_emb_params}")
    print(f"Total parameters: {total_params}")
    
    print("Percentage of non-embedding parameters: {:.2f}%".format(100 * non_emb_params / total_params))

import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    name: str
    target_params: float  # Target params in billions
    dim: int  # Model dimension
    n_heads: int  # Number of heads
    n_layers: int  # Number of layers
    
    @property
    def actual_params(self) -> float:
        """Calculate actual number of parameters (in billions) for the model"""
        model = load_model(self.dim, self.n_layers, self.n_heads)
        total_params = get_num_params(model, non_embedding=False)
        return total_params / 1e9
    
    @property
    def params_match(self) -> bool:
        """Check if actual params are within 10% of target"""
        return abs(self.actual_params - self.target_params) / self.target_params < 0.1

def create_model_configs() -> List[ModelConfig]:
    return [
        ModelConfig("52M", 0.052, 320, 5, 8),
        ModelConfig("0.1B", 0.102, 512, 8, 12),
        ModelConfig("0.15B", 0.153, 640, 10, 14),
        ModelConfig("0.21B", 0.212, 768, 12, 16),
        ModelConfig("0.28B", 0.284, 896, 14, 18),
        ModelConfig("0.37B", 0.373, 1024, 16, 20),
        ModelConfig("0.69B", 0.693, 1344, 21, 24),
    ]

def validate_configs(configs: List[ModelConfig]) -> pd.DataFrame:
    rows = []
    for config in configs:
        actual_params = config.actual_params
        error_pct = (actual_params - config.target_params) / config.target_params * 100
        
        rows.append({
            "Name": config.name,
            "Target N (B)": f"{config.target_params:.3f}",
            "Actual N (B)": f"{actual_params:.3f}",
            "Error %": f"{error_pct:+.1f}%",
            "dm": config.dim,
            "nh": config.n_heads,
            "L": config.n_layers,
        })
    
    return pd.DataFrame(rows)

def main():
    configs = create_model_configs()
    df = validate_configs(configs)
    print("\nModel Configurations and Parameter Counts:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    # test_init()
    main()
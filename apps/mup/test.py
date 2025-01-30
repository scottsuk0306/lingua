from apps.mup.transformer import BaseTransformer, BaseTransformerArgs


if __name__ == "__main__":
    args = BaseTransformerArgs(
        dim=256,
        n_layers=2,
        n_heads=8,
    )
    
    model = BaseTransformer(args)
    print(model)
from apps.mup.blocks import BaseTransformer, BaseTransformerArgs


def test_init():
    args = BaseTransformerArgs(
        dim=256,
        n_layers=2,
        n_heads=8,
    )
    
    model = BaseTransformer(args)
    print(model)
    
    # BaseTransformer(
    #   (rope_embeddings): RotaryEmbedding()
    #   (layers): ModuleList(
    #     (0-1): 2 x TransformerBlock(
    #       (attention): Attention(
    #         (wq): Linear(in_features=256, out_features=256, bias=False)
    #         (wk): Linear(in_features=256, out_features=256, bias=False)
    #         (wv): Linear(in_features=256, out_features=256, bias=False)
    #         (wo): Linear(in_features=256, out_features=256, bias=False)
    #       )
    #       (feed_forward): FeedForward(
    #         (w1): Linear(in_features=256, out_features=768, bias=False)
    #         (w3): Linear(in_features=256, out_features=768, bias=False)
    #         (w2): Linear(in_features=768, out_features=256, bias=False)
    #       )
    #       (attention_norm): RMSNorm()
    #       (ffn_norm): RMSNorm()
    #     )
    #   )
    # )
    

if __name__ == "__main__":
    test_init()
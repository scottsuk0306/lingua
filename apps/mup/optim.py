import logging
from typing import Any, Dict

from lingua.optim import OptimArgs, build_lr_fn
from torch import nn
from torch.optim import AdamW, lr_scheduler

logger = logging.getLogger()


def adjust_hidden_dim(hidden_dim, ffn_dim_multiplier, multiple_of):
    assert ffn_dim_multiplier is not None
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class MuPAdamW(AdamW):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        fused=False,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            fused=fused,
        )

        self.base_lr = lr

        for group in self.param_groups:
            if "lr_scale" not in group:
                group["lr_scale"] = 1.0

    def step(self, closure=None):
        for group in self.param_groups:
            group["lr"] = self.base_lr * group["lr_scale"]

        loss = super().step(closure=closure)

        return loss


def build_mup_adamw(model: nn.Module, args: OptimArgs, n_steps: int):
    logger.info("Starting build of muP AdamW optimizer...")

    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    matrix_like_decay_params = []
    matrix_like_hidden_decay_params = []
    vector_like_decay_params = []
    nodecay_params = []

    for n, p in param_dict.items():
        if p.dim() >= 2:
            if ("attention.w" in n and n.endswith(".weight")) or (
                "feed_forward.w" in n and n.endswith(".weight")
            ):
                if "wo" in n:
                    matrix_like_decay_params.append(p)
                elif "w2" in n:
                    matrix_like_hidden_decay_params.append(p)
                else:
                    matrix_like_decay_params.append(p)
            else:
                vector_like_decay_params.append(p)
        else:
            nodecay_params.append(p)

    hidden_dim = adjust_hidden_dim(
        model.config.dim, model.config.ffn_dim_multiplier, model.config.multiple_of
    )
    base_hidden_dim = adjust_hidden_dim(
        model.config.mup_dim_model_base,
        model.config.ffn_dim_multiplier,
        model.config.multiple_of,
    )
    in_width_multiplier = (
        float(model.config.dim / model.config.mup_dim_model_base) ** -1.0
    )
    hidden_width_multiplier = float(hidden_dim / base_hidden_dim) ** -1.0

    optim_groups = [
        {
            "params": matrix_like_decay_params,
            "weight_decay": args.weight_decay,
            "lr_scale": in_width_multiplier,
        },
        {
            "params": matrix_like_hidden_decay_params,
            "weight_decay": args.weight_decay,
            "lr_scale": hidden_width_multiplier,
        },
        {
            "params": vector_like_decay_params,
            "weight_decay": args.weight_decay,
            "lr_scale": 1.0,
        },
        {"params": nodecay_params, "weight_decay": 0.0, "lr_scale": 1.0},
    ]

    num_matrix_like_decay_params = sum(p.numel() for p in matrix_like_decay_params)
    num_matrix_like_hidden_decay_params = sum(
        p.numel() for p in matrix_like_hidden_decay_params
    )
    num_vector_like_decay_params = sum(p.numel() for p in vector_like_decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    logger.info(
        f"num matrix-like decayed parameter tensors: {len(matrix_like_decay_params)}, with {num_matrix_like_decay_params:,} parameters (lr scale: {in_width_multiplier})"
    )
    logger.info(
        f"num matrix-like hidden decayed parameter tensors: {len(matrix_like_hidden_decay_params)}, with {num_matrix_like_hidden_decay_params:,} parameters (lr scale: {hidden_width_multiplier})"
    )
    logger.info(
        f"num vector-like decayed parameter tensors: {len(vector_like_decay_params)}, with {num_vector_like_decay_params:,} parameters (lr scale: 1)"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters (lr scale: 1)"
    )

    optimizer = MuPAdamW(
        optim_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.epsilon,
        fused=True,
    )

    lr_fn = build_lr_fn(args, n_steps)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_fn)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler

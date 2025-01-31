import logging
from torch import nn
from torch.optim import AdamW, lr_scheduler

from lingua.optim import OptimArgs, build_lr_fn

logger = logging.getLogger()


### Begin muP code ### 
def build_mup_adamw(model: nn.Module, args: OptimArgs, n_steps: int, mup_factor: float):
    logger.info("Starting build of mup AdamW optimizer...")
    
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    mup_decay_params = []
    decay_params = []
    nodecay_params = []
    
    for n, p in param_dict.items():
        if p.dim() >= 2:
            if ('attention.w' in n and n.endswith('.weight')) or ('feed_forward.w' in n and n.endswith('.weight')):
                mup_decay_params.append(p)
            else:
                decay_params.append(p)
        else:
            nodecay_params.append(p)
    
    optim_groups = [
        {'params': mup_decay_params, 'weight_decay': args.weight_decay, 'lr_scale': 1 / mup_factor},
        {'params': decay_params, 'weight_decay': args.weight_decay, 'lr_scale': 1},
        {'params': nodecay_params, 'weight_decay': 0.0, 'lr_scale': 1}
    ]
    
    num_mup_decay_params = sum(p.numel() for p in mup_decay_params)
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num mup decayed parameter tensors: {len(mup_decay_params)}, with {num_mup_decay_params:,} parameters")
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    optimizer = AdamW(
        optim_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.epsilon,
        fused=True,  # Faster optim.step but can throw errors
    )

    # scheduler
    lr_fn = build_lr_fn(args, n_steps)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_fn
    )  # lr_scheduler.LambdaLR(optimizer, lr_fn)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
### End muP code ### 
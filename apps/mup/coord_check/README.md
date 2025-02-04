## Logs

## MuP v1
- Initial version

## MuP v2
- Added initialization for Feed forward and Attention layers

- `TransformerBlock.init_weights` change `use_mup=False` to `use_mup=True`

- Result: Step 0 became stable for all widths

## MuP v3

- Added learning rate scaling for different parameter groups

- Using the functionalities in `apps/mup/optim.py`

- Result: Stable hidden weights acitvation for larger widths (>512), still logit explodes

## MuP v4

- Add logit scaling before norm

## MuP v5

- Add logit scaling after norm

- Result: logit scaled properly

## MuP v6

- output layer init 0

- Result: Much more stable hidden weights and logits for smaller widths

## MuP v7

- Add qk_norm
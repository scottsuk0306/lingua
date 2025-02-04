## Logs

## MuP v1
- Initial version

## MuP v2
- Added initialization for Feed forward and Attention layers

- `TransformerBlock.init_weights` change `use_mup=False` to `use_mup=True`

- Step 0 became stable for all widths

## MuP v3

 - Added learning rate scaling for different parameter groups

 - Using the functionalities in `apps/mup/optim.py`

 
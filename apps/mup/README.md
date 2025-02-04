# MuP implementation details

| Operation | Specific Embedding | Details |
|-----------|-------------------|----------|
| Output Scaling | Multiply the output of the embedding by `scale_emb` | [view code](blocks.py#L587)
| Residual Connection Scaling | Scale the output tensor of a block before adding to each residual connection in each layer by `scale_depth/√num_layers` | [view code](blocks.py#L584-L594)
| Initialization of Tensors | Set the initialization standard deviation of each two-dimensional tensor parameter to `init_std/√(dm/dbase)`, and set other parameters' initialization to `init_std` | [view code](blocks.py#L599-L612) |
| Learning Rate Scaling of Tensors | Adjust the learning rate of each two-dimensional tensor parameter to `1/(dm/dbase)` times the learning rate of other parts (or the overall learning rate) |
| LM Head Scaling | Adjust the output logits to `1/(dm/dbase)` times the original value |


## 1. Output Scaling

## 2. Residual Connection Scaling

## 3. Initialization of Tensors

## 4. Learning Rate Scaling of Tensors

## 5. LM Head Scaling
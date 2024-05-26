## Tensor decompositions

Modified _torch.nn.Conv2d_ class using tensor decomposition techniques to compress weights and speed up layer inference.

To apply layer compression, you need to replace all layers of the model with the proposed layer and call the method _.decompose_. There is no need to change checkpoints.

Installation:
```
git clone <current repo>
pip install -e .
```

There are 3 approaches to decompose model weights:
```python
in_ch, out_ch = 64, 128
in_cr, out_sr = [32, 48] # core_ranks
st = 32 # stick_rank
ker_h, ker_w = 3, 5
conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size=(ker_h, ker_w))

# №1 (mode=1)
decomposed_layer = nn.Sequential(
    nn.Conv2d(in_ch,  in_cr,  kernel_size=(1, 1)),
    nn.Conv2d(in_cr,  out_sr, kernel_size=(ker_h, ker_w)),
    nn.Conv2d(out_sr, out_ch, kernel_size=(1, 1))
)

# №2 (mode=2) ---recomended---
decomposed_layer = nn.Sequential(
    nn.Conv2d(in_ch,  st,  kernel_size=(ker_h, 1)),
    nn.Conv2d(st,  out_ch, kernel_size=(1, ker_w))
)

# №3 (mode=3)
decomposed_layer = nn.Sequential(
    nn.Conv2d(in_ch,  in_cr,  kernel_size=(1, 1)),
    nn.Conv2d(in_ch,  st,     kernel_size=(ker_h, 1)),
    nn.Conv2d(st,     out_ch, kernel_size=(1, ker_w)),
    nn.Conv2d(out_sr, out_ch, kernel_size=(1, 1))
)
```

**Examples:**
Decomposed parameters are defined automatically.
```python
from td import Conv2dTD

conv_layer = Conv2dTD(64, 128, kernel_size=(3, 3))
conv_layer.decompose(mode=2)
```
With defined decomposed parameters.
```python
from td import Conv2dTD

conv_layer = Conv2dTD(64, 128, kernel_size=(3, 3), core_ranks=[32, 32], stick_rank=32)
conv_layer.decompose(mode=2)
```

To initialize the model with decomposed parameters, you need to export the environment variable with the previously chosen mode.
```
export IS_DECOMPOSED=2 
```
Then, all decomposed weights will be loaded correctly from the checkpoint.

For more details, look at the example in the _example.ipynb_.
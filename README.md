# Transformer Architecture Implementation

This repository contains an implementation of transformer architecture that was introduced by Vaswani et al. [1].  The Transformer is a neural network architecture that has been widely used in natural language processing tasks, such as machine translation and language modeling.

## Overview
The code in this repository implements the Transformer architecture using the PyTorch library. 
The implementation includes the core components of the Transformer, such as the multi-head self-attention mechanism and the position-wise feedforward networks.

## Usage 

```python
import sys        

local_x_transformers_module_path = './transformers'
if local_x_transformers_module_path not in sys.path:
    sys.path.append(local_x_transformers_module_path)
```

```python
from x_transformers import BaseDecoderGenerator, Batch, subsequent_mask

model = BaseDecoderGenerator(N, d_model, d_ff, vocab_size, heads, dropout, max_len)
```

## References
[1]	A. Vaswani et al., “Attention Is All You Need.” arXiv, Aug. 01, 2023. doi: 10.48550/arXiv.1706.03762.

## Credit

The code in this repository is based on the annotated version of the Transformer paper available at [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). The annotated version provides a line-by-line implementation of the Transformer architecture and has been a valuable resource for understanding the details of the model.



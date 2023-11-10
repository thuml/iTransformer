# iTransformer with Efficient Attentions

Since the attention mechanism is applied on the variate dimension in the inverted structure, efficient attentions with reduced complexity essentially addresses the problem of numerous variates, which is ubiquitous in real-world applications.

We currently try out the linear complexity attention from [Flowformer](https://github.com/thuml/Flowformer), and hardware-friendly attention mechanism from [FlashAttention](https://github.com/shreyansh26/FlashAttention-PyTorch). It demonstrates the efficiency improvement by adopting booming attention mechanisms.

## Results


**Environments**: The batch size of training is 16, and the batch size of inference is 1. The experiments run on P100 (16G).

|Traffic Dataset (862 Variates) |Transformer|Flowformer|FlashAttention|
|-|-|-|-|
|Training Speed (s/iter)|0.592 |**0.099** | **0.128**|
|Inference Speed (s/iter)| 0.097 | **0.095** | **0.096** |
|Training Memo (GiB)| 14.057 | **1.596** |**1.657** |
|Inference Memo (GiB)| 1.017 | **0.977** | **0.977** |

We are look forward to more efficient attention mechanisms for better multivariate correlation. Feel free to contact us!
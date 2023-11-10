# iTransformer for Variate Generalization

This folder contains the implementation of the iTransformer to generalize on unseen variates. If you are new to this repo, we recommend you to read this [README](../multivariate_forecast/README.md) first.

## Scripts

> During Training
<p align="center">
<img src="../../figures/pt.png" alt="" align=center />
</p>

> During Inference
<p align="center">
<img src="../../figures/pi.png" alt="" align=center />
</p>

In each folder named after the dataset, we provide two strategies to enable Transformers to generalize on unseen variate during inference. We use partial variates of the dataset for training, and test the obtained model directly on all variates.

* **Channel-Independence**: During training, the model regards each variate of multivariate time series as independent channels, and uses a shared backbone to forecast all channels. Therefore, the model can work for arbitrary variate channels one by one, but the training procedure can also be time-consuming.

* **Inverted Attention**: Benefiting from the flexibility of the attention mechanism that the number of input tokens can be dynamically changeable. In our Inverted Transformer the amount of **variates as tokens** is no longer restricted and thus feasible to vary from training and inference.

## Results

<p align="center">
<img src="../../figures/generability.png" alt="" align=center />
</p>

Performance of generalization on unseen variates. iTransformers can be naturally trained with 20% variates and accomplish forecast on all variates with hardly increased error, while Transformers with Channel Independence bring about significantly increased error.
# Inverted Transformers Work Better for Time Series Forecasting

This folder contains the comparison of the vanilla Transformer-based forecasters and the inverted versions. If you are new to this repo, we recommend you to read this [README](../multivariate_forecast/README.md) first.

## Scripts

In each folder named after the dataset, we provide the iTransformers and the vanilla Transformers experiments.

```
# iTransformer on the Traffic Dataset with gradually enlarged lookback windows.

bash ./scripts/boost_performance/Traffic/iTransformer.sh
```

You can change the ```model_name``` in the script to switch the selection of the vanilla Transformer and inverted version.

## Results

<p align="center">
<img src="../../figures/boosting.png" alt="" align=center />
</p>

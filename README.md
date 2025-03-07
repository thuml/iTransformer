# RxiTransformer

The source code is almost the same as that of iTransformer.

The main changes are described below.

# Model: RxiTransformer

RxiTransformer is a machine learning model for time series forecasting that combines elements of RLinear(https://github.com/plumprc/RTSF) with iTransformer(https://github.com/thuml/iTransformer). 


# Learnign Rate Scheduler: ReduceLRReverter

# Minor Changes

-exp_basic.py is revised to use added models.

-exp_reverter.py is added for impelmentation of ReduceLRReverter

-run_reverter.py is added for impelmentation of ReduceLRReverter

-requirements.txt is revised and torch is installed by torch.txt
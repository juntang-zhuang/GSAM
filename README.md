<h1 align="center"><b>GSAM Optimizer</b></h1>
<h3 align="center"><b>Surrogate Gap Guided Sharpness-Aware Minimization </b></h3>
[Paper, ICLR 2022](https://openreview.net/pdf?id=edONMAnhLu-) 

Disclaimer: original code (during internship at Google) was in jax and tensorflow and is planned to be released with Keras. This repository is a re-implmentation in PyTorch tested only on a Cifar10 experiment, not tested by reproduction of results in the paper

## Experiments

| Optimizer             | Test error rate |
| :-------------------- |   -----: |
| SGD + momentum        |   3.20 % |
| SAM + SGD + momentum  |   2.86 % |
| ASAM + SGD + momentum |   2.55 % |
| GSAM + SGD + momentum |   2.45 % | 

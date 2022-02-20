# DGL & Pytorch implementation of Enhanced Graph Embedding with Side information (EGES)

## Version
dgl==0.6.1, torch==1.9.0

## Paper
Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba: https://arxiv.org/abs/1803.02349

## Dataset
https://wx.jdcloud.com/market/jdata/list/17

## How to run
Run command: `python main.py` with default configuration, and the following message will shown up:

```
Using backend: pytorch
Num skus: 33344, num brands: 3662, num shops: 4785, num cates: 79
Epoch 00000 | Step 00000 | Step Loss 0.9117 | Epoch Avg Loss: 0.9117
Epoch 00000 | Step 00100 | Step Loss 0.8736 | Epoch Avg Loss: 0.8801
Epoch 00000 | Step 00200 | Step Loss 0.8975 | Epoch Avg Loss: 0.8785
Evaluate link prediction AUC: 0.6864
Epoch 00001 | Step 00000 | Step Loss 0.8695 | Epoch Avg Loss: 0.8695
Epoch 00001 | Step 00100 | Step Loss 0.8290 | Epoch Avg Loss: 0.8643
Epoch 00001 | Step 00200 | Step Loss 0.8012 | Epoch Avg Loss: 0.8604
Evaluate link prediction AUC: 0.6875
Epoch 00002 | Step 00000 | Step Loss 0.8659 | Epoch Avg Loss: 0.8659
Epoch 00002 | Step 00100 | Step Loss 0.8825 | Epoch Avg Loss: 0.8515
Epoch 00002 | Step 00200 | Step Loss 0.8644 | Epoch Avg Loss: 0.8482
Evaluate link prediction AUC: 0.6885
Epoch 00003 | Step 00000 | Step Loss 0.8370 | Epoch Avg Loss: 0.8370
Epoch 00003 | Step 00100 | Step Loss 0.8343 | Epoch Avg Loss: 0.8378
Epoch 00003 | Step 00200 | Step Loss 0.8189 | Epoch Avg Loss: 0.8352
Evaluate link prediction AUC: 0.6895
Epoch 00004 | Step 00000 | Step Loss 0.8211 | Epoch Avg Loss: 0.8211
Epoch 00004 | Step 00100 | Step Loss 0.8175 | Epoch Avg Loss: 0.8259
Epoch 00004 | Step 00200 | Step Loss 0.8287 | Epoch Avg Loss: 0.8234
Evaluate link prediction AUC: 0.6904
```

The AUC of link-prediction task on test graph is computed after each epoch is done.

## Reference
https://github.com/nonva/eges

https://github.com/wangzhegeek/EGES.git

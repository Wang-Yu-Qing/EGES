# DGL & Pytorch implementation of Enhanced Graph Embedding with Side information (EGES)

## Version
dgl==0.6.1, torch==1.9.0

## Paper
Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba: 

https://arxiv.org/pdf/1803.02349.pdf

https://arxiv.org/abs/1803.02349

## Dataset
https://wx.jdcloud.com/market/jdata/list/17

## How to run
Create folder named `data`. Download two csv files from [here](https://github.com/Wang-Yu-Qing/dgl_data/tree/master/eges_data) into the `data` folder.

Run command: `python main.py` with default configuration, and the following message will shown up:

```
Using backend: pytorch
Num skus: 1006, num brands: 221, num shops: 308, num cates: 55
Epoch 00000 | Step 00000 | Step Loss 0.8452 | Epoch Avg Loss: 0.8452
Evaluate link prediction AUC: 0.5557
Epoch 00001 | Step 00000 | Step Loss 0.7293 | Epoch Avg Loss: 0.7293
Evaluate link prediction AUC: 0.5774
Epoch 00002 | Step 00000 | Step Loss 0.7157 | Epoch Avg Loss: 0.7157
Evaluate link prediction AUC: 0.5764
...
Epoch 00028 | Step 00000 | Step Loss 0.7105 | Epoch Avg Loss: 0.7105
Evaluate link prediction AUC: 0.5880
Epoch 00029 | Step 00000 | Step Loss 0.7115 | Epoch Avg Loss: 0.7115
Evaluate link prediction AUC: 0.5914
```

The AUC of link-prediction task on test graph is computed after each epoch is done.

## Reference
https://github.com/nonva/eges

https://github.com/wangzhegeek/EGES.git

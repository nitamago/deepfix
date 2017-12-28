#! /bin/bash

# どのfoldをデータセットとして使うか
Fold_Num=0
Checkpoint=10
Batch=1
Emb=1

# RNN_CELLの中のユニット数
Mem=1
Layers=2

# 一つのデータを何回学習に利用するか
Epochs=10

# 学習をどのチェックポイントから再開するか
Resume=0
Resume_epoch=0
Resume_minibatch=0
RNN_cell=GRU

# ドロップアウト率
dropout=0.2

echo "./neural_net/train.py $Fold_Num $Batch $Emb $Mem $Layers $Epochs $Resume $Resume_epoch $Resume_minibatch $RNN_cell $Checkpoint $dropout"
./neural_net/train.py $Fold_Num $Batch $Emb $Mem $Layers $Epochs $Resume $Resume_epoch $Resume_minibatch $RNN_cell $Checkpoint $dropout

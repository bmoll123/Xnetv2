## Environment

```
conda create --name xnet python=3.10
conda activate xnet
pip install -r requirements.txt
```
## Dataset

資料集擺放
-XNetv2
-Glas

## Training
```
python train_glas_semi.py \
  --data_root ../Glas \
  --portion unsegSplit20 \
  --exp_name my_exp \
  -e 200 -b 2 --val_interval 1 \
  --confidence_threshold 0.5
```

## Goal
Model,Iou ↑,Dice ↑,ASD ↓,95HD ↓
MT,76.41,86.62,2.65,13.28
EM,76.81,86.88,2.54,12.28
UAMT,76.55,86.72,2.73,13.43
CCT,77.60,87.39,2.27,11.23
CPS,80.46,89.17,2.08,10.56
URPC,76.84,86.91,2.31,10.97
CT,79.02,88.28,2.33,12.02
XNet,80.89,89.44,2.07,9.86
XNet v2,83.17,90.81,1.75,8.54
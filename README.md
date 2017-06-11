# Faster RCNN

To run this code, you need to install mxnet and other dependencies first. To do that, you can run
```bash
bash script/additional_deps.sh
```
to install MXNet and other required packages of python.

Then, you need to run
```bash
bash script/get_voc.sh
bash script/get_pretrained_model.sh
```
to get VOC data and pretrained VGG-16 network.

To download the KITTI dataset you need to request the download link, hence we do not provide a
script here. The data should have the structure like
```
├── kitti
│   ├── images
│   ├── imglists
│   └── results
```

Our own data is not public, hence we can only present the result. We can not release the dataset.

## VOC
To train on the VOC-07 data using pretrained VGG-16, run
```bash
python train_end2end.py --image_set 2007_trainval --gpu 0
```

To use both VOC-07 and VOC-12 data, run
```bash
python train_end2end.py --image_set 2007_trainval+2012_trainval --gpu 0
```

After train completed, run
```bash
python test.py --gpu 0
```
to test on VOC-07 test dataset.


## KITTI
To train on the KITTI data using model pretrained on VOC,
uncomment
```python
# del arg_params['cls_score_weight'], arg_params['cls_score_bias']
# del arg_params['bbox_pred_weight'], arg_params['bbox_pred_bias']
```
in `load_param` function in `rcnn/utils/load_model.py`

Then, run
```bash
python train_end2end.py --dataset Kitti --pretrained e2e --pretrained_epoch 10 --prefix e2e_kitti
```

After train completed, run 
```bash
python test.py --dataset Kitti --image_set test --gpu 0 --prefix e2e_kitti --epoch 10 --thresh 0.01
```
then submit the result to KITTI website to get the result.

## Own Dataset
Since the data is not public yet, we do not provide the script to train/test it. However it is
very similar to the KITTI dataset.

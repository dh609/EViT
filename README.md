# EViT
## Folder Structure

Prepare the following folders to organize this repo:
```none
EViT
├── EViT (code)
├── model_weights
├── results (save the masks predicted by models)
├── lightning_logs (training logs)
├── data
│   ├── whubuilding
│   │   ├── train
│   │   │   ├── images (original images)
│   │   │   ├── masks_origin (original masks)
│   │   │   ├── masks (converted masks)
│   │   ├── val (the same with train)
│   │   ├── test (the same with test)
│   │   ├── train_val (Merge train and val)
│   ├── spacenet
│   │   ├── train
│   │   │   ├── images (original images)
│   │   │   ├── masks_origin (original masks)
│   │   │   ├── masks (converted masks)
│   │   ├── val (the same with train)
│   │   ├── test (the same with test)
│   │   ├── train_val (Merge train and val)
```

## Datasets
 * [WHU](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html "WHU")
 * [SPACENET](https://spacenet.ai/spacenet-buildings-dataset-v2/ "SPACENET")

## Usage
 Clone the repository: git clone https://github.com/dh609/EViT.git  
   Hyper-parameters configuration and training are implemented in config/whubuilding/evit.py;  
   test.py predict the test dataset;
   Hyper-parameters configuration andtesting are implemented in config/whubuilding/evit.py.

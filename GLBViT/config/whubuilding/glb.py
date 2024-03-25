from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.whubuilding_dataset import *

from geoseg.models.GLBViT import GLVBiT
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 110
ignore_index = 255
train_batch_size = 4
val_batch_size = 4
lr = 1e-4
weight_decay = 0.0025
backbone_lr = 1e-4
backbone_weight_decay = 0.0025
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES
#final2   max + avg
weights_name = ""
weights_path = "model_weights/whubuilding/{}".format(weights_name)
test_weights_name = ""
log_name = 'whubuilding/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [1]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None
#  define the network
net = GLBViT(num_classes=num_classes)

# define the loss
loss = EdgeLoss(ignore_index=255)
use_aux_loss = False

# define the dataloader

 train_dataset = WHUBuildingDataset(data_root='data/whubuilding/train_val', mode='train', mosaic_ratio=0.25, transform=train_aug)
 val_dataset = WHUBuildingDataset(data_root='data/whubuilding/val', mode='val', transform=val_aug)
 test_dataset = WHUBuildingDataset(data_root='data/whubuilding/test', mode='val', transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)


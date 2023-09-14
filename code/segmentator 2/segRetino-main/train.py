import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
import glob
from tqdm import tqdm
# from segretino.unet import UNET
from segretino.unet_model import UNet
from segretino.loss import DiceLoss, IoU
from segretino.training_utils import Train, Evaluate

from data.dataset import DriveDataset

# train_x = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/imgs/train/'+'*.jpg'))
# train_y = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/masks/train/'+'*.jpg'))
#
# valid_x = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/imgs/val/'+'*.jpg'))
# valid_y = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/masks/val/'+'*.jpg'))

# train_x = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/seg2/t_train_1/images/'+'*.jpg'))
# train_y = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/seg2/t_train_1/1st_manual/'+'*.jpg'))
#
# valid_x = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/seg2/t_test_1/images/'+'*.jpg'))
# valid_y = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/seg2/t_test_1/mask/'+'*.jpg'))

train_x = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/imgs3/train/'+'*.jpg'))
train_y = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/masks3/train/'+'*.jpg'))

valid_x = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/imgs3/val/'+'*.jpg'))
valid_y = sorted(glob.glob('F:/Nova/RetinaImageSynthesisProject/Code/pytorch-CycleGAN-and-pix2pix/segmentator/data/masks3/val/'+'*.jpg'))


# Dataloader
s = 0.2
train_dataset = DriveDataset(train_x, train_y, scale=s)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)
val_dataset = DriveDataset(valid_x, valid_y, scale=s)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# Training and Evaluate object
train = Train(dice=DiceLoss(), iou=IoU())
eval = Evaluate(dice=DiceLoss(), iou=IoU())

# Model Initialization and setting up hyperparameters
model = UNet(n_channels=3, n_classes=1, bilinear=False)
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 100

# Training
for epoch in tqdm(range(epochs)):
    print("Epoch: ", epoch)

    train_dice, train_iou = train.forward(model=model, loader=train_loader, optimizer=optimizer)
    val_dice, val_iou = eval.forward(model=model, loader=val_loader, optimizer=optimizer)

torch.save(model.state_dict(), 'unet_new3.pth')

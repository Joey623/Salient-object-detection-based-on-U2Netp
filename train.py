import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import numpy as np
import glob

from dataloader import Rescale
from dataloader import RescaleT
from dataloader import RandomCrop
from dataloader import ToTensor
from dataloader import ToTensorLab
from dataloader import SalObjDataset
from model import UNet_Light

""" define loss function"""
# Binary Cross Entropy
# bce_loss = nn.BCELoss(size_average=True)
bce_loss = nn.BCELoss(reduction='mean')
# Dice Loss
def dice_loss(input, targets):
    N = targets.size()[0]
    smooth = 1
    input_flat = input.view(N, -1)
    targets_flat = targets.view(N, -1)
    intersection = input_flat * targets_flat
    dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
    loss = 1 - dice_eff.sum() / N
    return loss

# Fusion Loss
def fusion_loss(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss0 = bce_loss(d0, labels_v)
    bce_loss1 = bce_loss(d1, labels_v)
    bce_loss2 = bce_loss(d2, labels_v)
    bce_loss3 = bce_loss(d3, labels_v)
    bce_loss4 = bce_loss(d4, labels_v)
    bce_loss5 = bce_loss(d5, labels_v)
    bce_loss6 = bce_loss(d6, labels_v)

    dice_loss0 = dice_loss(d0, labels_v)
    dice_loss1 = dice_loss(d1, labels_v)
    dice_loss2 = dice_loss(d2, labels_v)
    dice_loss3 = dice_loss(d3, labels_v)
    dice_loss4 = dice_loss(d4, labels_v)
    dice_loss5 = dice_loss(d5, labels_v)
    dice_loss6 = dice_loss(d6, labels_v)

    loss1 = bce_loss1 + dice_loss1
    loss2 = bce_loss2 + dice_loss2
    loss3 = bce_loss3 + dice_loss3
    loss4 = bce_loss4 + dice_loss4
    loss5 = bce_loss5 + dice_loss5
    loss6 = bce_loss6 + dice_loss6

    loss_side = 0.3 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.1 * loss4 + 0.1 * loss5 + 0.1 * loss6
    loss = bce_loss0 + dice_loss0
    loss_all = loss + loss_side
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" %
          (loss.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss, loss_all

# set the directory of training dataset

model_name = 'unet_light'
data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'im_aug' + os.sep)
tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'gt_aug' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)


epoch_num = 400
batch_size_train = 1   # default=16
train_num = 0


tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tra_lbl_name_list = []

for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]
    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)


print("-"*30)
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))

print("-"*30)

train_num = len(tra_img_name_list)


# pretreatment
salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)])
)


salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)  # default num_workers=1


# define the model(UNet_Light)
print("model:UNet_Light")
net = UNet_Light(3, 1)
if torch.cuda.is_available():
    net.cuda()

# define optimizer(Use Adam)
print("---define optimizer---")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


# start training
print("---start training---")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000  # save the .pth file every 2000 iterations
Loss = []
tar_Loss = []


for epoch in range(0, epoch_num):
    net.train()
    train_epoch_loss = []

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = fusion_loss(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()
        train_epoch_loss.append(loss.data.item())

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f" %(
          epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val
        ))

        Loss.append(running_loss / ite_num4val)
        tar_Loss.append(running_tar_loss / ite_num4val)
        if epoch % 10 == 9:
            if (epoch + 1) >= 200 and (epoch + 1) <= 400:
                torch.save(net.state_dict(), model_dir + "unetlight{}.pth".format(epoch+1))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()
            ite_num4val = 0
    train_epoch_loss.append(np.average(train_epoch_loss))

Loss0 = torch.tensor(Loss)
tar_Loss0 = torch.tensor(tar_Loss)
train_epoch_loss0 = torch.tensor(train_epoch_loss)
np.save('epoch{}_train_loss'.format(epoch_num), Loss0)
np.save('epoch{}_tar'.format(epoch_num), tar_Loss0)
np.save('train_epoch{}_loss'.format(epoch_num), train_epoch_loss0)










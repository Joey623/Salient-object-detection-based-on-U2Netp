# Salient-object-detection-based-on-U2Netp
## Requirements
torch, PIL, matplotlib, opencv, seaborn.
```
pip install torch
pip install pillow
pip install matplotlib
pip install opencv-python
pip install seaborn
```
## Training
You can run "train.py", but it needs training dataset. In this project we use DUTS-TR.
```
python train.py
```
## Application
It needs the network weights. You can find it in the saved_models folder.
```
python utils.py
```
## Acknowledgement
This project is based on U2Net created by Qin.X _et al._
[original project address](https://github.com/xuebinqin/U-2-Net).

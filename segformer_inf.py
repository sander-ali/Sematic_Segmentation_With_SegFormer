from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import torch
from torch import nn
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import albumentations as aug
import matplotlib.pyplot as plt
import cv2

""" For Inference """
model_inf = torch.load('model/segformermodel.pt')
df = pd.read_csv('drone_dataset/class_dict_seg.csv')
cat = df['name']
pal = df[[' r', ' g', ' b']].values
id2cat = cat.to_dict()
cat2id = {s: t for t, s in id2cat.items()}

rdir = 'drone_dataset'
feat_ext = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
#uncomment the following and comment the commend in line 156 if you have GPU device
#dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = torch.device("cpu")
model_inf = model_inf.to(dev)

img = Image.open('drone_dataset/images/test/564_5.jpg')
mask = Image.open('drone_dataset/mask/test/564_5.png').convert('L')

fig, axs = plt.subplots(1,2, figsize=(20,10))
axs[0].imshow(img)
axs[1].imshow(mask)
plt.show()

feat_ext_inf = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
pval = feat_ext_inf(img, return_tensors="pt").pixel_values.to(dev)

model_inf.eval()
out = model_inf(pixel_values = pval)
logs = out.logits.cpu()

uplogs = nn.functional.interpolate(logs, size = img.size[::-1], mode = 'bilinear', align_corners=False)
seg = uplogs.argmax(dim=1)[0]
col_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
for lab, col in enumerate(pal):
    col_seg[seg == lab, :] = col

col_seg = col_seg[..., ::-1]
img = np.array(img) * 0.5 + col_seg * 0.5
img = img.astype(np.uint8)
fig, axs = plt.subplots(1,2, figsize = (20,10))
axs[0].imshow(img)
axs[1].imshow(col_seg)
plt.show()


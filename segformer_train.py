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

w, h = 512, 512

class SegImg(Dataset):
    def __init__(self, rdir, feat_ext, transforms = None, train = True):
        self.rdir = rdir
        self.feat_ext = feat_ext
        self.train = train
        self.transforms = transforms
        
        spath = "train" if self.train else "test"
        self.idir = os.path.join(self.rdir, "images", spath)
        self.adir = os.path.join(self.rdir, "mask", spath)
        
        imgfiles = []
        for r, d, f in os.walk(self.idir):
            imgfiles.extend(f)
        self.images = sorted(imgfiles)
        
        annfiles = []
        for r, d, f in os.walk(self.adir):
            annfiles.extend(f)
        self.annotations = sorted(annfiles)
        
        assert len(self.images) == len(self.annotations) 
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        img = cv2.imread(os.path.join(self.idir, self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mapseg = cv2.imread(os.path.join(self.adir, self.annotations[idx]))
        mapseg = cv2.cvtColor(mapseg, cv2.COLOR_BGR2GRAY)
        
        if self.transforms is not None:
            augment = self.transforms(image=img, mask=mapseg)
            in_enc = self.feat_ext(augment['image'], augment['mask'], return_tensors="pt")
        else:
            in_enc = self.feat_ext(img, mapseg, return_tensors="pt")
        
        for s,t in in_enc.items():
            in_enc[s].squeeze_()
            
        return in_enc

tr = aug.Compose([aug.Flip(p=0.5)])

rdir = 'drone_dataset'
feat_ext = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
tr_data = SegImg(rdir = rdir, feat_ext = feat_ext, transforms = tr)
vl_data = SegImg(rdir = rdir, feat_ext = feat_ext, transforms = None, train = False)

print("Training Examples", len(tr_data))
print("Validation Examples", len(vl_data))

in_enc = tr_data[0]

mask = in_enc["labels"].numpy()

tr_dloader = DataLoader(tr_data, batch_size=8, shuffle=True)
vl_dloader = DataLoader(vl_data, batch_size=8)

batch = next(iter(tr_dloader))

cat = pd.read_csv('drone_dataset/class_dict_seg.csv')['name']
id2cat = cat.to_dict()
cat2id = {t: s for s, t in id2cat.items()}

mdl = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True, 
                                                       num_labels = len(id2cat), id2label = id2cat, label2id = cat2id,
                                                       reshape_last_stage=True)

opt = AdamW(mdl.parameters(), lr = 0.00002)
#uncomment the following and comment the commend in line 92 if you have GPU device
#dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = torch.device("cpu")
mdl.to(dev)
print("Initialization Completed")

for ep in range(1,15):
    print("Epoch Number", ep)
    sl = tqdm(tr_dloader)
    tr_acc = []
    tr_loss = []
    vl_acc = []
    vl_loss = []
    mdl.train()
    for idx, batch in enumerate(sl):
        pval = batch["pixel_values"].to(dev)
        gt = batch["labels"].to(dev)
        opt.zero_grad()
        out = mdl(pixel_values = pval, labels = gt)
        
        uplog = nn.functional.interpolate(out.logits, size = gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = uplog.argmax(dim=1)
        
        mask = (gt != 255)
        pred_class = pred[mask].detach().cpu().numpy()
        gt_class = gt[mask].detach().cpu().numpy()
        acc = accuracy_score(pred_class, gt_class)
        loss = out.loss
        tr_acc.append(acc)
        tr_loss.append(loss.item())
        sl.set_postfix({'Batch': idx, 'Pixel-wise acc': sum(tr_acc)/len(tr_acc), 'loss': sum(tr_loss)/len(tr_loss)})
        loss.backward()
        opt.step()
  
    print(f"Pixel-wise Train Accuracy: {sum(tr_acc)/len(tr_acc)}\
          Train Loss: {sum(tr_loss)/len(tr_loss)}\
              Pixel-wise Validation Accuracy: {sum(vl_acc)/len(vl_acc)}\
                  Validation Loss: {sum(vl_loss)/len(vl_loss)}")

torch.save(mdl, 'model/segformermodel.pt')


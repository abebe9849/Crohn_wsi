##

import hydra
from omegaconf import DictConfig, OmegaConf
import sys,gc,os,random,time,math
import matplotlib.pyplot as plt
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter
from  torch.cuda.amp import autocast, GradScaler 
import timm
print(timm.__version__)
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp
import pydicom
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from functools import partial
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,f1_score,log_loss
from  sklearn.metrics import accuracy_score as acc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau,CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from albumentations import *
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip,RandomGamma, RandomRotate90,GaussNoise,Cutout,RandomBrightnessContrast,RandomContrast,RandomCrop
from albumentations.pytorch import ToTensorV2
import logging


## https://github.com/abebe9849/code_factory
from code_factory.pooling import GeM,AdaptiveConcatPool2d
from code_factory.loss_func import SmoothCrossEntropy,MyCrossEntropyLoss
from code_factory.augmix import RandomAugMix
from code_factory.gridmask import GridMask
#from code_factory.sam import SAMSGD
from code_factory.radam import RAdam
from code_factory.fmix import sample_mask
from code_factory.loss_func import SmoothCrossEntropy,MyCrossEntropyLoss
from code_factory.cam import *

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    x = F.relu(x)
    b, c, h, w = x.size()
    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))
    
    return F.relu(x - epsilon) / (max_value + epsilon)

def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    if shape is not None:
        h, w, c = shape
        cam = cv2.resize(cam, (w, h))
    cam = cv2.applyColorMap(cam, mode)
    return cam
def denormalize(image, mean=np.array([0.485, 0.456, 0.406]),std=np.array([0.229, 0.224, 0.225]), dtype=np.uint8, tp=True):
    if tp:
        image = image.transpose((1, 2, 0))#imgaeはnumpy 
    if mean is not None:
        image = (image * std) + mean
    if dtype == np.uint8:
        image *= 255.
        return image.astype(np.uint8)
    else:
        return image

def time_function_execution(function_to_execute):
    def compute_execution_time(*args, **kwargs):
        start_time = time.time()
        result = function_to_execute(*args, **kwargs)
        end_time = time.time()
        computation_time = end_time - start_time
        print('Computation lasted: {}'.format(computation_time))
        return result
    return compute_execution_time

def time_function(function_to_execute):
    def compute_execution_time(*args, **kwargs):
        start_time = time.time()
        result = function_to_execute(*args, **kwargs)
        end_time = time.time()
        computation_time = end_time - start_time
        print('model: {}'.format(computation_time))
        return result
    return compute_execution_time

class TrainDataset(Dataset):
    def __init__(self, df,CFG,train=True,transform1=None, transform2=None):
        self.df = df
        self.transform = transform1
        self.transform_ = transform2
        self.CFG = CFG
        self.train = train

    def __len__(self):
        return len(self.df)
    #@time_function_execution
    def __getitem__(self, idx):
        file_path = self.df["file_png"].values[idx]
        #print(file_path)
        image = cv2.imread(file_path)
        image = cv2.resize(image,(self.CFG.preprocess.size,self.CFG.preprocess.size))
        label_ = self.df["label"].values[idx]
        
        if self.CFG.preprocess.diff and label_==1:
            diff = np.ones_like(image)*np.array([7,8,9])
            diff = diff.astype("uint8")
            #print(diff.dtype,image.dtype)
            image +=diff
        if self.transform:
            image = self.transform(image=image)['image']
            """
            if idx<1000:
                if image.dtype=="float64":
                    image = image.astype("float32")
                print(image.dtype)
                cv2.imwrite(f"aug_image{idx}.png",image)
            """
        if self.transform_:
            image = self.transform_(image=image)['image']
        
        label = torch.tensor(label_).long()
        onehot_label = torch.eye(self.CFG.model.n_classes)[label]
        return image, label,onehot_label,torch.tensor(idx)

class TestDataset(Dataset):
    def __init__(self, df,CFG,train=True,transform1=None, transform2=None):
        self.df = df
        self.transform = transform1
        self.transform_ = transform2
        self.CFG = CFG
        self.train = train

    def __len__(self):
        return len(self.df)
    #@time_function_execution
    def __getitem__(self, idx):
        file_path = self.df["file_png"].values[idx]
        image = cv2.imread(file_path)
        image = cv2.resize(image,(self.CFG.preprocess.size,self.CFG.preprocess.size))
        label_ = self.df["label"].values[idx]
        if self.CFG.preprocess.diff and label_==1:
            diff = np.ones_like(image)*np.array([7,8,9])
            diff = diff.astype("uint8")
            image +=diff
        if self.transform:
            image = self.transform(image=image)['image']
        if self.transform_:
            image = self.transform_(image=image)['image']
        
        label = torch.tensor(label_).long()
        onehot_label = torch.eye(self.CFG.model.n_classes)[label]
        return image, label#,torch.tensor(idx)

class TrainDataset_(Dataset):
    #tileにした4枚を返す
    def __init__(self, df,CFG,train=True,transform1=None, transform2=None):
        self.df = df
        self.transform = transform1
        self.transform_ = transform2
        self.CFG = CFG
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file'].values[idx]
        image = cv2.imread(f"/home/u094724e/ダウンロード/byori/cam/data/{file_path}_6.png")
        image = cv2.resize(image,(self.CFG.preprocess.size,self.CFG.preprocess.size))
        label_ = self.df["label"].values[idx]
        if self.transform:
            image = self.transform(image=image)['image']
 
        if self.transform_:
            image = self.transform_(image=image)['image']
        
        label = torch.tensor(label_).long()
        onehot_label = torch.eye(self.CFG.model.n_classes)[label]
        return image, label,onehot_label,torch.tensor(idx)

SEQ_POOLING = {
    'gem': GeM(dim=2),
    'concat': AdaptiveConcatPool2d(),
    'avg': nn.AdaptiveAvgPool2d(1),
    'max': nn.AdaptiveMaxPool2d(1)
}

class Model(nn.Module):
    def __init__(self,CFG, num_classes=2, base_model='tf_efficientnet_b0_ns',pool="avg",pretrain=True):
        super(Model, self).__init__()
        self.base_model = base_model #"str"
        self.CFG = CFG
        self.model = timm.create_model(self.base_model, pretrained=pretrain, num_classes=10)
        nc = self.model.num_features
        self.cam = nn.Conv2d(nc,num_classes,1)
        if pool in ('avg','concat','gem','max'):
            self.avgpool = SEQ_POOLING[pool]
            if pool == "concat":
                nc *= 2
        self.last_linear = nn.Linear(nc,num_classes)
    #@time_function
    def forward(self, input1):#0.04066038131713867...1024*1024,bs=4
        x = self.model.forward_features(input1)
        cam_feature = self.cam(x)
        feature = self.avgpool(x).view(input1.size()[0], -1)
        y = self.last_linear(feature)
        return y,feature#,cam_feature


def get_transforms1(*, data,CFG):
    if data == 'train':
        return Compose([
            #RandomCrop(512,512,p=1),
            HorizontalFlip(p=CFG.augmentation.hflip_p),
            VerticalFlip(p=CFG.augmentation.vflip_p),
            RandomContrast(p=CFG.augmentation.contrast_p),
            #GaussNoise(p=0.5),
            RandomRotate90(p=CFG.augmentation.rotate_90_p),
            #RandomGamma(p=0.5),
            RandomBrightnessContrast(p=CFG.augmentation.bright_contrast_p),
            RandomAugMix(severity=CFG.augmentation.augmix_s, width=3, alpha=1., p=CFG.augmentation.augmix_p),
            #GaussianBlur(p=0.5),
            GridMask(num_grid=CFG.augmentation.grdimask_n, p=CFG.augmentation.grdimask_p),
            Cutout(p=CFG.augmentation.cutout_p,max_h_size=CFG.augmentation.cutout_h,max_w_size=CFG.augmentation.cutout_w),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    elif data == 'valid':
        return Compose([Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],)])
def to_tensor(*args):
    return Compose([ToTensorV2()])

from  sklearn.metrics import roc_auc_score,accuracy_score
def AUC(y_true,y_pred):
    auc = roc_auc_score(y_true,y_pred)
    return auc

def train_fn(CFG,fold,folds):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"### fold: {fold} ###")
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True),train=True, 
                                 transform1=get_transforms1(data='train',CFG=CFG),transform2=to_tensor(),CFG=CFG)#
    valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True),train=False,
                                 transform1=get_transforms1(data='valid',CFG=CFG),transform2=to_tensor(),CFG=CFG)#

    train_loader = DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size, shuffle=False, num_workers=4)


    ###  model select ============
    if CFG.model.type=="cnn":
        model = Model(num_classes=CFG.model.n_classes,base_model=CFG.model.name,pool=CFG.model.pooling,CFG=CFG)
    #elif CFG.model.type=="vit":
    #    model = Model_vit(num_classes=CFG.model.n_classes,base_model=CFG.model.name,CFG=CFG)
    model.to(device)
    # ============

    ###  optim select ============
    if CFG.train.optim=="adam":
        optimizer = Adam(model.parameters(), lr=CFG.train.lr, amsgrad=False)
    elif CFG.train.optim=="radam":
        optimizer = RAdam(model.parameters(), lr=CFG.train.lr)
    #elif CFG.train.optim=="sam":
    #    raise ValueError("another .py")
    #    optimizer = SAMSGD(model.parameters(), lr=CFG.train.lr, rho=0.05)
    # ============

    ###  scheduler select ============
    if CFG.train.scheduler.name=="cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.train.epochs, eta_min=CFG.train.scheduler.min_lr)
    elif CFG.train.scheduler.name=="cosine_warm":
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=CFG.train.scheduler.t_0, T_mult=1, eta_min=CFG.train.scheduler.min_lr, last_epoch=-1)
    elif CFG.train.scheduler.name=="reduce":
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)
    # ============
    
    ###  loss select ============
    if CFG.augmentation.mix_p>0:
        criterion = SmoothCrossEntropy(smoothing =CFG.loss.smooth_a,one_hotted=True)
    elif CFG.loss.name=="ce" and CFG.loss.weights==None:
        criterion = SmoothCrossEntropy(smoothing =CFG.loss.smooth_a)
    elif CFG.loss.name=="ce" and CFG.loss.weights!=None:
        criterion = MyCrossEntropyLoss(weight=CFG.loss.weights)
    elif CFG.loss.name=="focal" and CFG.loss.weights!=None:
        criterion = FocalLoss_CE(gamma=CFG.loss.focal_gamma)
    elif CFG.loss.name=="focal_cosine":
        criterion = FocalCosineLoss(gamma=CFG.loss.focal_gamma)
    # ============
    #metric_fc = AdaCos(num_features=1280, num_classes=CFG.model.n_classes).cuda()
    
    softmax = nn.Softmax(dim = 1)
    scaler = torch.cuda.amp.GradScaler()
    best_score = 0
    best_loss = np.inf
    best_preds = None
    
    
    for epoch in range(CFG.train.epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels,onehot_label,indexes) in tk0:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            onehot_label = onehot_label.to(device)

            if CFG.train.amp:
                with autocast():
                    y_preds,_ = model(images.float())
                    loss = criterion(y_preds, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                y_preds,_ = model(images.float())
                loss = criterion(y_preds, labels)
                loss.backward()
                optimizer.step()
        if CFG.train.scheduler.name!="none":
            scheduler.step()


            avg_loss += loss.item() / len(train_loader)
        model.eval()
        avg_val_loss = 0.
        preds = []
        valid_labels = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images, labels,onehot_label,indexes) in tk1:
            images = images.to(device)
            labels = labels.to(device)
            onehot_label = onehot_label.to(device)
            if CFG.train.amp_inf:
                with torch.no_grad():
                    with autocast():
                        y_preds,_ = model(images.float())
                        loss = criterion(y_preds,labels)
            else:
                with torch.no_grad():
                    y_preds,_ = model(images.float())
                    loss = criterion(y_preds, labels)
                    mask = onehot_label.unsqueeze(2).unsqueeze(3)
            valid_labels.append(labels.to('cpu').numpy())
            softmax = nn.Softmax(dim = 1)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy())
            
            avg_val_loss += loss.item() / len(valid_loader)
        preds = np.concatenate(preds)
        valid_labels = np.concatenate(valid_labels)

        print(preds.shape,valid_labels.shape)

        score = log_loss(valid_labels,preds)
        auc_score = AUC(valid_labels,preds[:,1])
        threshed_preds = np.argmax(preds, axis=1)
        acc_acore = accuracy_score(valid_labels,threshed_preds)


        elapsed = time.time() - start_time
        log.info(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f}  time: {elapsed:.0f}s')
        log.info(f'  Epoch {epoch+1} - AUC mean: {auc_score:.6f}')
        log.info(f'  Epoch {epoch+1} - Acc : {acc_acore:.4f}')
        if auc_score>best_score:#AUCのスコアが良かったら予測値を更新...best_epochをきめるため
            best_score = auc_score
            best_preds = preds
            log.info(f'  Epoch {epoch+1} - Save Best ＡＵＣ: {best_score:.4f}')
            torch.save(model.state_dict(), f'fold{fold}_{CFG.general.exp_num}_bestloss.pth')
    return best_preds, valid_labels

def inference_cam(model, test_loader,device,CFG,test):
    cams = []
    _features = []
    scaler = torch.cuda.amp.GradScaler()
    cam_type =None
    cam_type = "grad"

    for i, (images,labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        print(f"true_label:{labels.data}")
        if cam_type!="grad":
            with torch.no_grad():
                cam, idx= model(images)
        else:
            cam, idx= model(images)
        cam = cam.to('cpu')
        images = images.to('cpu')
        images = reverse_normalize(images)  
        print(cam.size())
        c = (cam.squeeze().numpy())
        plt.imshow(c)    
        plt.savefig(f"test_cam{i}.png",cmap="jet")    

        heatmap = visualize(images, cam)
        print(heatmap.shape)
        hm = (heatmap.squeeze().numpy().transpose(1, 2, 0))#[..., ::-1]
        images = images.squeeze().numpy().transpose(1, 2, 0)#[..., ::-1]
        plt.imshow(images)
        plt.savefig(f"test_img{i}.png")
        plt.imshow(hm)
        plt.savefig(f"test_heat{i}.png")


    cams = np.concatenate(cams)
    return cams

def inference(model, test_loader,device,CFG,test):
    preds = []
    scaler = torch.cuda.amp.GradScaler()

    for i, (images,labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            y_preds,_ = model(images.float())
            softmax = nn.Softmax(dim = 1)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy())

    preds = np.concatenate(preds)
    return preds

def submit(test,CFG):
        print('run inference')
        torch.cuda.set_device(CFG.general.device)
        get_cam = False
        if get_cam:
            CFG.train.batch_size=1


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_dataset = TestDataset(test,transform1=get_transforms1(data='valid',CFG=CFG),transform2=to_tensor(),CFG=CFG)
        test_loader = DataLoader(test_dataset, batch_size=CFG.train.batch_size, shuffle=False)
        probs = []
        features = []
        for fold in range(5):
            weights_path =f'fold{fold}_{CFG.general.exp_num}_bestloss.pth'
            #weights_path =f"/home/u094724e/ダウンロード/byori/cam/outputs/2021-02-24/10-20-43/fold{fold}_000_bestloss.pth"
            if CFG.model.type=="cnn":
                model = Model(num_classes=CFG.model.n_classes,base_model=CFG.model.name,pool=CFG.model.pooling,CFG=CFG)
            elif CFG.model.type=="vit":
                model = Model_vit(num_classes=CFG.model.n_classes,base_model=CFG.model.name,CFG=CFG)
            state_dict = torch.load(weights_path,map_location=device)
            model.load_state_dict(state_dict)
            if get_cam:
                target_layer = model.model.conv_head
            model.to(device)
            model.eval()

            if get_cam:
                #model = CAM(model, target_layer)
                #model = ScoreCAM(model, target_layer)
                model =GradCAM(model, target_layer)
                _cam = inference_cam(model, test_loader, device,CFG,test)
                print(type(_cam),_cam.shape)
            if CFG.tta.do:
                _probs = inference_tta(model, test_loader, device,CFG)
            else:
                _probs = inference(model, test_loader, device,CFG,test)
            probs.append(_probs)
            #features.append(_features)
        probs = np.mean(probs, axis=0)
        #features = np.mean(features, axis=0)
        return probs#,features

log = logging.getLogger(__name__)
CONFIG_path='config/tile_conf.yaml'
@hydra.main(config_path=CONFIG_path)
def main(CFG : DictConfig) -> None:
    #CFG = OmegaConf.to_yaml(cfg)

    seed_torch(seed=42)
    log.info(f"===============exp_num{CFG.general.exp_num}============")
    
    test = pd.read_csv("test_tile_croped.csv")
    if CFG.general.debug:
        folds = pd.read_csv("folds_debug_tile_croped.csv")
    else:
        folds = pd.read_csv("folds_tile_croped.csv")

    if CFG.loss.weights==0:
        CFG.loss.weights=None

    print(folds.shape)
    print(folds["fold"].value_counts())


    if CFG.general.precrop:
        test = pd.read_csv("test_tile_croped_new.csv")
        folds = pd.read_csv("folds_tile_croped_new.csv")
        print(folds.shape)
        print(folds["fold"].value_counts())


    
    preds = []
    valid_labels = []
    for fold in range(5):
        _preds, _valid_labels = train_fn(CFG,fold,folds)
        print(_preds.shape,_valid_labels.shape)
        print(AUC(_valid_labels,_preds[:,1]))
        preds.append(_preds)
        valid_labels.append(_valid_labels)
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    logloss_score = log_loss(valid_labels,preds)
    auc_score = AUC(valid_labels,preds[:,1])
    threshed_preds = np.argmax(preds, axis=1)
    print(threshed_preds.shape,valid_labels.shape,valid_labels[:10])
    acc_acore = accuracy_score(valid_labels,threshed_preds)

    log.info(f'  =====logloss(CV)====== {logloss_score}')
    log.info(f'  =====AUC(CV)====== {auc_score}')
    log.info(f'  =====Acc(CV)====== {acc_acore}')

    #test = test.sample(len(test)//5).reset_index(drop=True)

    pred = submit(test,CFG)
    logloss_test = log_loss(test['label'].values[:],pred)
    auc_test = AUC(test["label"].values[:],pred[:,1])
    threshed_pred = np.argmax(pred, axis=1)
    acc_test = accuracy_score(test["label"].values[:],threshed_pred)

    log.info(f'  =====logloss(inner_test)====== {logloss_test}')
    log.info(f'  =====AUC(inner_test)====== {auc_test}')
    log.info(f'  =====Acc(inner_test)====== {acc_test}')

    ##
    for i in range(CFG.model.n_classes):
        col = f"pred_{i}"
        test[col]=pred[:,i]
    test.to_csv("predict.csv")

    

if __name__ == "__main__":
    main()

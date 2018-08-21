# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 20:19:23 2018

@author: Allen
"""

import zipfile
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as ply
import os
import sys
import imageio
from PIL import Image
import glob
import matplotlib.pyplot as plt
import time
import math
import datetime as dt
import pytz
import pickle
import logging
from io import BytesIO


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
else:    
    dtype = torch.FloatTensor
    
    
class IOU_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y):
        #print(y_pred.requires_grad)
        #y_pred = torch.where(y_pred.ge(0.5), torch.tensor(1.0), torch.tensor(0.0))
        i = y_pred.mul(y)
        u = (y_pred + y) - i
        mean_iou = torch.mean(i.view(i.shape[0],-1).sum(1) / u.view(i.shape[0],-1).sum(1))
        iou_loss = 1 - mean_iou
        #from boxx import g
        #g()
        
        return iou_loss
      

        
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (int): Desired output size. 
    """

    def __init__(self, scale='random', min_scale=1, max_scale=3):
        self.scale = scale 
        self.min_scale = min_scale   
        self.max_scale = max_scale   

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if self.scale == 'random':
            current_scale = np.clip((np.random.rand() * self.max_scale), self.min_scale, self.max_scale)
        else:
            current_scale = self.scale
      
        output_size = round(np.max(image.shape) * current_scale)        
        
        if mask is not None:
            image = np.concatenate([image,mask],2)
        resized_img = transform.resize(image, (output_size, output_size), mode='constant', preserve_range=True)
        #print(resized_img.shape)
        img_final = resized_img[:,:,0:1]
        if mask is not None:
            mask_final = resized_img[:,:,1:]

        return {'image':img_final, 'mask':mask_final}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (int): Desired output size. 
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if mask is not None:
            image = np.concatenate([image,mask],2)

        h, w = image.shape[:2]

        new_h = new_w = self.output_size
        top = 0 if h == new_h else np.random.randint(0, h - new_h)
        left = 0 if w == new_w else np.random.randint(0, w - new_w)


        cropped_image = image[top: top + new_h,
                      left: left + new_w]
        
        img_final = cropped_image[:,:,0:1]
        if mask is not None:
            mask_final = cropped_image[:,:,1:]

        return {'image':img_final, 'mask':mask_final}

class Flip(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (int): Desired output size. 
    """

    def __init__(self, orient='random'):
        assert orient in ['H', 'V', 'NA', 'random']
        self.orient = orient

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if self.orient=='random':
            current_orient = np.random.choice(['H', 'W', 'NA', 'NA'])     
        else:
            current_orient = self.orient
        
        if mask is not None:
            image = np.concatenate([image,mask],2)

        if current_orient == 'H':
            flipped_image = image[:,::-1,:] - np.zeros_like(image)
        elif current_orient == 'W':
            flipped_image = image[::-1,:,:] - np.zeros_like(image)
        else:
            # do not flip if orient is NA
            flipped_image = image
        img_final = flipped_image[:,:,0:1]
        if mask is not None:
            mask_final = flipped_image[:,:,1:]

        return {'image':img_final, 'mask':mask_final}

'''composed = transforms.Compose([Rescale(scale='random', max_scale=5),
                               RandomCrop(101),
                               Flip(orient='random')])


transformed = composed({'image':image, 'mask':mask})
x_final, m_final = transformed['image'], transformed['mask']'''

class SaltDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, np_img, np_mask, df_depth, mean_img, img_out_size=101, transform=None):
        """
        Args:
            data_dir (string): Path to the image files.
            train (bool): Load train or test data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.np_img = np_img
        self.np_mask = np_mask.clip(0,1)
        self.df_depth = df_depth
        self.mean_img = mean_img
        self.img_out_size = img_out_size
        self.transform = transform

    def __len__(self):
        return len(self.np_img)

    def __getitem__(self, idx):

        X_orig = self.np_img[idx]
        X = X_orig - self.mean_img
        
        if self.np_mask is None:
            y = np.zeros((101,101,1))
        else:
            y = self.np_mask[idx]
            
        if self.transform:
            transformed = self.transform({'image':X, 'mask': y})
            X = transformed['image']
            y = transformed['mask']
            
        #print(X.dtype)
        X = np.moveaxis(X, -1,0)
        
        pad_size = self.img_out_size - self.np_img.shape[2]
        X = np.pad(X, [(0, 0),(0, pad_size), (0, pad_size)], mode='constant')
        #print(X.dtype)

        d = self.df_depth.iloc[idx,0]
        #id = self.df_depth.index[idx]
        #from boxx import g
        #g()
        X = torch.from_numpy(X).float().type(dtype)
        y = torch.from_numpy(y).float().squeeze().type(dtype)

        return (X,y,d,idx)
      

class SaltNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1,64,3, padding=10),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        out = self.seq(X)
        return torch.clamp(out[:,:,:-1,:-1].squeeze(), 0.0, 1.0)
    
    
def load_all_data():
    try:
        print('Try loading data from npy and pickle files...')
        np_train_all = np.load('./data/np_train_all.npy')
        np_train_all_mask = np.load('./data/np_train_all_mask.npy')
        np_test = np.concatenate([np.load('./data/np_test_0.npy'), np.load('./data/np_test_1.npy')])
        with open('./data/misc_data.pickle', 'rb') as f:
            misc_data = pickle.load(f)
        print('Data loaded.')
        return (np_train_all, np_train_all_mask, np_test, misc_data)
    
    except:
        print('npy files not found. Reload data from raw images...')
        np_train_all, np_train_all_ids = load_img_to_np('./data/train/images')
        np_train_all_mask, np_train_all_mask_ids = load_img_to_np('./data/train/masks')
        df_train_all_depth = pd.read_csv('./data/depths.csv').set_index('id')
        np_test, np_test_ids = load_img_to_np('./data/test/images')
        np.save('./data/np_train_all.npy', np_train_all)
        np.save('./data/np_train_all_mask.npy', np_train_all_mask)
        for k, v in enumerate(np.split(np_test,2)):
            np.save(f'./data/np_test_{k}.npy', v)
        misc_data = {'df_train_all_depth': df_train_all_depth,
                     'np_train_all_ids': np_train_all_ids,
                     'np_train_all_mask_ids': np_train_all_mask_ids,
                     'np_test_ids': np_test_ids}
        with open('./data/misc_data.pickle', 'wb') as f:
            pickle.dump(misc_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data loaded.')
        return (np_train_all, np_train_all_mask, np_test, misc_data)
          
  
def rle_encoder2d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    s = pd.Series(x.clip(0,1).flatten('F'))
    s.index = s.index+1
    df = s.to_frame('pred').assign(zero_cumcnt=s.eq(0).cumsum())
    df = df.loc[df.pred.gt(0)]
    df_rle = df.reset_index().groupby('zero_cumcnt').agg({'index': min, 'pred': sum}).astype(int).astype(str)
    rle = ' '.join((df_rle['index'] + ' '+df_rle['pred']).tolist())
    
    return rle
  
  
def rle_encoder3d(x):   
    return np.r_[[rle_encoder2d(e) for e in x]]
  
  
def load_img_to_np(img_path, num_channel=1):
    images = []
    img_ids = []
    for filename in sorted(glob.glob(f'{img_path}/*.png')): #assuming png
        img_id = filename.split('\\')[-1].split('.')[0]
        img_ids.append(img_id)
        images.append(np.array(imageio.imread(filename), dtype=np.uint8).reshape(101,101,-1)[:,:,0:num_channel])
    return (np.r_[images], img_ids)
  
  
def load_single_img(path, show=False):
    img = np.array(imageio.imread(path), dtype=np.uint8)
    if show:
        plt.imshow(img, cmap='gray')
    return img
  
  
def calc_raw_iou(a, b):
    if isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().detach().numpy()
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    u = np.sum(np.clip(a+b, 0, 1), (1,2)).astype(np.float)
    i = np.sum(np.where((a+b)==2, 1, 0), (1,2)).astype(np.float)
    with np.errstate(divide='ignore',invalid='ignore'):
        iou = np.where(i==u, 1, np.where(u==0, 0, i/u))
        
    return iou
  
  
def calc_mean_iou(a, b):
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    iou = calc_raw_iou(a, b)
    iou_mean = (iou[:,None]>thresholds).mean(1).mean()

    return iou_mean
  
  
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
  
  
def get_current_time_as_fname():
        timestamp = (
                dt.datetime.now(pytz.timezone('Australia/Melbourne'))
                .strftime('%Y_%m_%d_%H_%M_%S')
                )
                
        return timestamp
      
      
def plot_img_mask_pred(images, labels=None, img_per_line=8):
    images = [i.cpu().detach().numpy().squeeze() if isinstance(i, torch.Tensor) else i.squeeze() for i in images]
    num_img = len(images)
    if labels is None:
        labels = range(num_img)

    rows = np.ceil(num_img/img_per_line).astype(int)
    cols = min(img_per_line, num_img)
    f, axarr = plt.subplots(rows,cols)
    if rows==1:
        axarr = axarr.reshape(1,-1)
    f.set_figheight(3*min(img_per_line, num_img)//cols*rows)
    f.set_figwidth(3*min(img_per_line, num_img))
    for i in range(num_img):
        r = i//img_per_line
        c = np.mod(i,img_per_line)
        axarr[r,c].imshow(images[i], cmap='gray')
        axarr[r,c].grid()
        axarr[r,c].set_title(labels[i])

    plt.show()
    
    
def adjust_predictions(zero_mask_cut_off, X, y_pred, y=None):
    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    y_pred_adj = y_pred.clip(0,1)

    # Set predictions to all 0 for black images
    black_img_mask = (X.mean((1,2,3)) == 0)
    y_pred_adj[black_img_mask]=0

    # set all predictions to 0 if the number of positive predictions is less than ZERO_MASK_CUTOFF
    y_pred_adj = np.r_[[e if e.sum()>zero_mask_cut_off else np.zeros_like(e) for e in y_pred_adj]]
    
    if y is not None:
        print(f'IOU score before: {calc_mean_iou(y_pred, y)}, IOU Score after:{calc_mean_iou(y_pred_adj, y)}')
        
    return y_pred_adj
  
def show_img_grid():
    pass
    #plt.imshow(torchvision.utils.make_grid(torch.from_numpy(y_train_black).unsqueeze(1)).permute(1, 2, 0))
    
    
def join_files(filePrefix, filePath, newFileName=None, returnFileObject=False, removeChunks=False):  
    noOfChunks = int(glob.glob(f'{filePath}/{filePrefix}*')[0].split('-')[-1])
    dataList = []
    j = 0
    for i in range(0, noOfChunks, 1):
        j += 1
        chunkName = f"{filePrefix}-chunk-{j}-Of-{noOfChunks}"
        f = open(chunkName, 'rb')
        dataList.append(f.read())
        f.close()
        if removeChunks:
            os.remove(chunkName)

    if returnFileObject:
        fileOut = BytesIO()
        for data in dataList:
            fileOut.write(data)
        fileOut.seek(0)
        return fileOut    
    else:
        fileOut = open(newFileName, 'wb')
        for data in dataList:
            fileOut.write(data)
        f2.close()
        print(f'File parts merged to {newFileName} successfully.')
    
# define the function to split the file into smaller chunks
def split_file_save(inputFile, outputFilePrefix, outputFolder, chunkSize=10000000):
    # read the contents of the file
    if isinstance(inputFile, BytesIO):
        data = inputFile.read()
        inputFile.close()
    else:
        f = open(inputFile, 'rb')
        data = f.read()
        f.close()

# get the length of data, ie size of the input file in bytes
    bytes = len(data)

# calculate the number of chunks to be created
    if sys.version_info.major == 3:
        noOfChunks = int(bytes / chunkSize)
    elif sys.version_info.major == 2:
        noOfChunks = bytes / chunkSize
    if(bytes % chunkSize):
        noOfChunks += 1

    chunkNames = []
    j = 0
    for i in range(0, bytes + 1, chunkSize):
        j += 1
        fn1 = f"{outputFilePrefix}-chunk-{j}-Of-{noOfChunks}"
        chunkNames.append(fn1)
        f = open(f'{outputFolder}/{fn1}', 'wb')
        f.write(data[i:i + chunkSize])
        f.close()
        
    return chunkNames
        
def get_logger(logger_name, level=logging.DEBUG):
    # logger
    file_name = '{}{}'.format('logs/',
                                logger_name)
    timestamp = dt.datetime.now(pytz.timezone('Australia/Melbourne'))\
        .strftime('%Y_%m_%d_%Hh')
    log_file = '{}_{}.log'.format(file_name, timestamp)
    logger = logging.getLogger(logger_name)

    formatter = (
        logging
        .Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   datefmt='%d/%m/%Y %H:%M:%S')
    )

    # for printing debug details
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(level)

    # for printing error messages
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.DEBUG)

    logger.setLevel(level)
    logger.handlers = []
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logging.getLogger(logger_name)
        
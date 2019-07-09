import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import math
import glob
from PIL import Image
import matplotlib.pyplot as plt

def one_hot(label, n_class, device):  
    # ラベルをOne-Hoe形式に変換
    eye = torch.eye(n_class, device=device)
    # ランダムベクトルあるいは画像と連結するために(B, c_class, 1, 1)のTensorにして戻す
    return eye[label].view(-1, n_class, 1, 1)  


def concat_image_label(image, label, n_class, device):
    # 画像とラベルを連結する
    B, C, H, W = image.shape    # 画像Tensorの大きさを取得    
    oh_label = one_hot(label, n_class, device)       # ラベルをOne-Hot形式に変換
    oh_label = oh_label.expand(B, n_class, H, W)  # ラベルを画像サイズに拡大
    return torch.cat((image, oh_label), dim=1)    # 画像とラベルをチャネル方向（dim=1）で連結



class make_dataset(utils.data.Dataset):
    def __init__(self, img_path, img_size):
        self.img_path = img_path + '*.jpg'
        self.images = glob.glob(self.img_path)
        self.transform = transforms.Compose(
            [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 画像読み込み
        image_name = self.images[idx]
        image = Image.open(image_name)
        image = image.convert('RGB') # PyTorch 0.4以降
        return self.transform(image)
    
    
    
def show_img(inputs,title='images'):
    """Imshow for Tensor."""
    inputs = torchvision.utils.make_grid(inputs[:8], nrow=8)
    inputs = inputs.detach().numpy().transpose((1,2,0))
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title(title)
    plt.imshow(inputs)
    
    

def make_mask(img_size, in_channels, p=0.5, s=(0.02, 0.1), r=(0.3, 3)):
    h, w = img_size
    # define mask
    mask = torch.ones([in_channels,w,h])
    
    # determine size of mask at random from the range of s(0,02, 0.4)times of the original image
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])

    # determine aspect ratio of mask at random from the range of r(0.3, 3)
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]

    # determine height and width of mask
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1
        
    # determine position of mask (x1,y1,x2,y2)
    x1 = np.random.randint(0, w - mask_width)
    y1 = np.random.randint(0, h - mask_height)
    x2 = x1 + mask_width
    y2 = y1 + mask_height
    mask[:, y1:y2, x1:x2] = 0
    
    return mask
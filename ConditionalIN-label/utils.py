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
import cv2

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
    inputs = torchvision.utils.make_grid(inputs, nrow=8)
    inputs = inputs.detach().numpy().transpose((1,2,0))
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title(title)
    plt.imshow(inputs)

def paint(images, thickness=1):
    color = (0,0,0)
    w,h = images.shape[2:]
    images_paint = []
    images = images.numpy()
    for image in images:
        image = image.transpose(1,2,0).copy()
        for _ in range(10):
            p1 = (np.random.randint(0,w-1), np.random.randint(0,h-1))
            p2 = (np.random.randint(0,w-1), np.random.randint(0,h-1))
            cv2.line(image, p1, p2, color, thickness=thickness)
        image = image.transpose(2,0,1)
        images_paint += [image]
    return torch.Tensor(images_paint)


def noise(images):
    noise = torch.rand(images.shape)
    return torch.add(images.cpu(), 0.3*noise)


def gaussian(images, kernel=(5,5)):
    images = images.numpy()
    images_gaussian = []
    for image in images:
        image = image.transpose(1,2,0)
        image = cv2.GaussianBlur(image,kernel,0)
        image = image.transpose(2,0,1)
        images_gaussian += [image]
    return torch.Tensor(images_gaussian)

def mosaic(images, ratio=0.3, area=None):
    # area=(x,y,width,height)
    if area==None: area = (0,0,images.shape[2],images.shape[3])
    x,y,width,height = area
    images_mosaic = []
    images = images.numpy()
    for image in images:
        image = image.transpose(1,2,0).copy()
        small = cv2.resize(image[y:y+height, x:x+width], None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(small, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        image = image.transpose(2,0,1)
        images_mosaic += [image]
    return torch.Tensor(images_mosaic)
        
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import json

from PIL import Image

# Modified from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', 
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def load_bbox_dict(filepath):
    with open(filepath) as data_file:    
        dic = json.load(data_file)
        return dic
    

def make_img_dataset(img_dir, bbox_dir, class_to_idx):
    images = []
    for target in os.listdir(img_dir):
        d = os.path.join(img_dir, target)
        # print("Processing: " + d)
        if not os.path.isdir(d):
            continue
        
        bbox_dict = load_bbox_dict(os.path.join(bbox_dir, target + ".json"))

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    label = [class_to_idx[target]]
                    label.extend(bbox_dict[fname])
                    item = (path, np.array(label).astype(int))
                    # print(item)
                    images.append(item)
                # break #TODO remove
            # break #TODO remove
        

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    return pil_loader(path)
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    """

class AttnDataset(data.Dataset):

    def __init__(self, img_root, bbox_root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(img_root)
        imgs = make_img_dataset(img_root, bbox_root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + img_root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.img_root = img_root # Root dir containing images
        self.bbox_root = bbox_root # Root dir containing bboxes
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        self.idx_to_class = {}
        for cls in self.class_to_idx:
            idx = self.class_to_idx[cls]
            self.idx_to_class[idx] = cls

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


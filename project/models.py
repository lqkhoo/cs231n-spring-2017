import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

import torchvision

def layer_inspector_hook(module, input, output):
    print(module)
    print("   " + str(output.size()))

def output_hook(module, input, output):
    # print(output)
    return output


class Upsample(nn.ConvTranspose2d):
    """
    Scales up any given 4D tensor along the spatial dimensions by the given factor.
    This is simply a transpose 2D conv with kernel and stride = factor, weights=1 and bias=0
    
    channels is the input channel. Output channels is also this
    """
    
    def __init__(self, channels, factor=1):
        fac = factor
        nn.ConvTranspose2d.__init__(self, channels, channels, kernel_size=fac, stride=fac)
        
        self.weight.data = torch.ones(channels,channels,fac,fac)
        self.bias.data = torch.zeros(1)
        for param in self.parameters():
            param.requires_grad = False
            
        

class MyModule(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
        
    def set_retrain(self, module_names, bool):
        for name in module_names:
            module = self._modules[name]
            for param in module.parameters():
                param.requires_grad = bool
                
    def freeze_weights(self):
        for module_name in self._modules.keys():
            module = self._modules[module_name]
            for param in module.parameters():
                param.requires_grad = False
                
                

class AuxNet(MyModule):
    """
    This is a simple conv-relu - conv-relu - fc neural net
    """
    
    def __init__(self, spatial_size, channels=1):
        MyModule.__init__(self)
        
        # No. of channels in conv layers
        self.CONV1_C = 32
        self.CONV2_C = 32
        
        self.spatial_size = spatial_size
        self.channels = channels
        
        # We don't pool because we want to preserve spatial information
        self.conv = nn.Sequential(
                nn.Conv2d(channels, self.CONV1_C, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=self.CONV1_C),
                nn.ReLU(),
                nn.Conv2d(self.CONV1_C, self.CONV2_C, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=self.CONV2_C),
                nn.ReLU()
                )
                
        self.fc = nn.Linear(self.CONV2_C * self.spatial_size * self.spatial_size, 4)
        
    def forward(self, x):
        
        x = self.conv.forward(x)
        x = x.view(-1, self.CONV2_C * self.spatial_size * self.spatial_size) # Flatten
        x = self.fc.forward(x)
        return x
        
    
    
class ResnetHost(MyModule):
    
    # Resnet consists of layers reading in input, and 4 residual 
    #   "blocks" called conv2_x to conv5_x
    # Then finally there's one avg pooling layer and a 512->1000 FC
    
    def __init__(self, n_classes=160):
        
        MyModule.__init__(self)

        resnet18 = torchvision.models.resnet18(pretrained=True)
        # Freeze all params
        for param in resnet18.parameters():
            param.requires_grad = False
            
        layers = list(resnet18.children())
        
        self.conv = layers[0]
        self.relu = layers[1]
        self.maxpool = layers[2]
        self.bn = layers[3]
        self.conv2_x = layers[4]
        self.conv3_x = layers[5]
        self.conv4_x = layers[6]
        self.conv5_x = layers[7]
        self.avg_pool = layers[8]
        # self.fc = layers[9]
        self.fc = nn.Linear(512, n_classes)
    
    
    def forward(self, x):
        
        x = self.conv.forward(x)
        x = self.relu.forward(x)
        x = self.maxpool.forward(x)
        x = self.bn.forward(x)
        x = self.conv2_x.forward(x)
        x = self.conv3_x.forward(x)
        x = self.conv4_x.forward(x)
        x = self.conv5_x.forward(x)
        x = self.avg_pool.forward(x)
        x = x.view(-1, 512) # Rather than 64, 512, which cannot account for partial minibatches
        x = self.fc.forward(x)
        
        return x
    

    
class AuxResNet(MyModule):
    
    # Parameters are (pre-trained) nets as we are doing transfer learning
    def __init__(self, resnethost, auxnet):
        MyModule.__init__(self)
        
        self.resnethost = resnethost
        self.auxnet = auxnet
        self.up2x = Upsample(1, factor=2)
        self.up4x = Upsample(1, factor=4)
    
    def forward(self, x):
        
        x = self.resnethost.conv.forward(x)
        x = self.resnethost.relu.forward(x)
        x = self.resnethost.maxpool.forward(x)
        x = self.resnethost.bn.forward(x)
        x = self.resnethost.conv2_x.forward(x) # 64 x 56 x 56
        x = self.resnethost.conv3_x.forward(x) # 128 x 28 x 28
        a1 = torch.mean(x, 1)
        x = self.resnethost.conv4_x.forward(x) # 256 x 14 x 14
        a2 = torch.mean(x, 1)
        x = self.resnethost.conv5_x.forward(x) # 512 x 7 x 7
        a3 = torch.mean(x, 1)
        x = self.resnethost.avg_pool.forward(x)
        x = x.view(-1, 512) # Rather than 64, 512, which cannot account for partial minibatches
        x = self.resnethost.fc.forward(x)
        
        a2 = self.up2x(a2)
        a3 = self.up4x(a3)
        
        a = torch.cat((a1, a2, a3), dim=1) # Concat along channels dimension
        a = self.auxnet.forward(a)
        scores = x
        bbox = a
        return (scores, bbox) # x is the image class (resnet's output)
                      # a is the bounding box (auxnet's output)
        
    
    # This is used for visualization
    def get_wouts(self, x, as_separate_channels=True):
        
        x = self.resnethost.conv.forward(x)
        x = self.resnethost.relu.forward(x)
        x = self.resnethost.maxpool.forward(x)
        x = self.resnethost.bn.forward(x)
        x = self.resnethost.conv2_x.forward(x) # 64 x 56 x 56
        x = self.resnethost.conv3_x.forward(x) # 128 x 28 x 28
        a1 = torch.mean(x, 1)
        x = self.resnethost.conv4_x.forward(x) # 256 x 14 x 14
        a2 = torch.mean(x, 1)
        x = self.resnethost.conv5_x.forward(x) # 512 x 7 x 7
        a3 = torch.mean(x, 1)
        x = self.resnethost.avg_pool.forward(x)
        x = x.view(-1, 512) # Rather than 64, 512, which cannot account for partial minibatches
        x = self.resnethost.fc.forward(x)
        
        a2 = self.up2x(a2)
        a3 = self.up4x(a3)
        
        # Note that we switch around the RGB channels
        if as_separate_channels == True:
            return (a2, a3, a1)
        else:
            return torch.cat((a2, a3, a1), dim=1)
    
    

class ModVggHost(MyModule):
    
    def __init__(self, n_classes=160):
        
        MyModule.__init__(self)
        
        # VGG consists of 2 sequential units, first contains conv-relu-pools
        # 2nd consists of FC's and dropouts
        
        vgg13 = torchvision.models.vgg13(pretrained=True)
        # Freeze all params
        for param in vgg13.parameters():
            param.requires_grad = False
        
        feat_net = list(vgg13.children())[0]
        feat_layers = list(feat_net.children())
        
        pool1 = feat_net[4] # 64 x 112 x 112
        pool2 = feat_net[9] # 128 x 56 x 56
        pool3 = feat_net[14] # 256 x 28 x 28
        pool4 = feat_net[19] # 512 x 14 x 14
        pool5 = feat_net[24] # 512 x 7 x 7
        
        self.feats1 = nn.Sequential(*(feat_layers[0:5]))
        self.feats2 = nn.Sequential(*(feat_layers[5:10])) # 128 x 52 x 52
        self.bn2 = nn.BatchNorm2d(128)
        self.feats3 = nn.Sequential(*(feat_layers[10:15])) # 256 x 28 x 28
        self.bn3 = nn.BatchNorm2d(256)
        self.feats4 = nn.Sequential(*(feat_layers[15:20])) # 512 x 14 x 14
        self.bn4 = nn.BatchNorm2d(512)
        self.feats5 = nn.Sequential(*(feat_layers[20:25])) # 512 x 7 x 7
        self.bn5 = nn.BatchNorm2d(512)
        
        classifier_net = list(vgg13.children())[1]
        classifier_layers = list(classifier_net.children())
        self.last = nn.Linear(4096, n_classes)
        self.classifier = nn.Sequential(*(classifier_layers[:-1]), self.last)
    
    def forward(self, x):
        
        x = self.feats1.forward(x)
        x = self.feats2.forward(x)
        x = self.bn2.forward(x)
        x = self.feats3.forward(x)
        x = self.bn3.forward(x)
        x = self.feats4.forward(x)
        x = self.bn4.forward(x)
        x = self.feats5.forward(x)
        x = self.bn5.forward(x)
        x = x.view(-1, 25088) # -1 is to account for partial minibatches
        x = self.classifier.forward(x)
        
        scores = x
        
        return scores


    
class AuxModVggHost(MyModule):
    
    def __init__(self, modvgghost, auxnet):
        MyModule.__init__(self)
        
        self.modvgghost = modvgghost
        self.auxnet = auxnet
        self.up2x = Upsample(1, factor=2)
        self.up4x = Upsample(1, factor=4)
    
    def forward(self, x):
        
        x = self.modvgghost.feats1.forward(x)
        x = self.modvgghost.feats2.forward(x)
        x = self.modvgghost.bn2.forward(x)
        x = self.modvgghost.feats3.forward(x)
        x = self.modvgghost.bn3.forward(x)
        a1 = torch.mean(x, 1)
        x = self.modvgghost.feats4.forward(x)
        x = self.modvgghost.bn4.forward(x)
        a2 = torch.mean(x, 2)
        x = self.modvgghost.feats5.forward(x)
        x = self.modvgghost.bn5.forward(x)
        a3 = torch.mean(x, 3)
        x = x.view(-1, 25088) # -1 is to account for partial minibatches
        x = self.modvgghost.classifier.forward(x)
        
        a2 = self.up2x(a2)
        a3 = self.up4x(a3)
        
        a = torch.cat((a1, a2, a3), dim=1) # Concat along channels dimension
        a = self.auxnet.forward(a)
        
        scores = x
        bbox = a
        
        return (scores, bbox)
    
    
    def get_wouts(self, x):
        
        x = self.modvgghost.feats1.forward(x)
        x = self.modvgghost.feats2.forward(x)
        x = self.modvgghost.bn2.forward(x)
        x = self.modvgghost.feats3.forward(x)
        x = self.modvgghost.bn3.forward(x)
        a1 = torch.mean(x, 1)
        x = self.modvgghost.feats4.forward(x)
        x = self.modvgghost.bn4.forward(x)
        a2 = torch.mean(x, 2)
        x = self.modvgghost.feats5.forward(x)
        x = self.modvgghost.bn5.forward(x)
        a3 = torch.mean(x, 3)
        
        a2 = self.up2x(a2)
        a3 = self.up4x(a3)
        
        # Note that we switch around the RGB channels
        if as_separate_channels == True:
            return (a2, a3, a1)
        else:
            return torch.cat((a2, a3, a1), dim=1)
        
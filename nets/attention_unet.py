import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

from .routeconv import RouteConv2D, RouteConvTranspose2D

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int, name):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            OrderedDict([
            (name+'_W_g'+'_0', RouteConv2D(F_g, F_int, name=name+'_W_g'+'_0', kernel_size=1,stride=1,padding=0,bias=True)),
            (name+'_W_g'+'_1', nn.BatchNorm2d(F_int))
            ])
            )
        
        self.W_x = nn.Sequential(
            OrderedDict([
            (name+'_W_x'+'_0', RouteConv2D(F_l, F_int, name=name+'_W_x'+'_0', kernel_size=1,stride=1,padding=0,bias=True)),
            (name+'_W_x'+'_1', nn.BatchNorm2d(F_int))
        ])
        )

        self.psi = nn.Sequential(
            OrderedDict([
            (name+'_W_psi'+'_0', RouteConv2D(F_int, 1, name=name+'_W_psi'+'_0', kernel_size=1,stride=1,padding=0,bias=True)),
            (name+'_W_psi'+'_1', nn.BatchNorm2d(1)),
            (name+'_W_psi'+'_2', nn.Sigmoid())
        ])
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class attention_UNet(nn.Module):

    def __init__(self, input_shape, in_channels=3, out_channels=2, init_features=32, untrack_bn=False):
        super(attention_UNet, self).__init__()
        if untrack_bn:
            bn_affine = False
            bn_track = False
        else:
            bn_affine = True
            bn_track = True

        features = init_features
        self.encoder1 = attention_UNet._block(in_channels, features, name="enc1", bn_affine=bn_affine, bn_track=bn_track, prefix_name="encoder1.")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = attention_UNet._block(features, features * 2, name="enc2", bn_affine=bn_affine, bn_track=bn_track, prefix_name="encoder2.")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = attention_UNet._block(features * 2, features * 4, name="enc3", bn_affine=bn_affine, bn_track=bn_track, prefix_name="encoder3.")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = attention_UNet._block(features * 4, features * 8, name="enc4", bn_affine=bn_affine, bn_track=bn_track, prefix_name="encoder4.")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = attention_UNet._block(features * 8, features * 16, name="bottleneck", bn_affine=bn_affine, bn_track=bn_track, prefix_name="bottleneck.")

        self.att4 = Attention_block(features * 8, features * 8, features * 4, name="att4")
        self.att3 = Attention_block(features * 4, features * 4, features * 2, name="att3")
        self.att2 = Attention_block(features * 2, features * 2, features, name="att2")
        self.att1 = Attention_block(features * 1, features * 1, features//2, name="att1")

        self.upconv4 = RouteConvTranspose2D(
            features * 16, features * 8, kernel_size=2, name="upconv4" ,stride=2
        )
        self.decoder4 = attention_UNet._block((features * 8) * 2, features * 8, name="dec4", bn_affine=bn_affine, bn_track=bn_track, prefix_name="decoder4.")

        self.upconv3 = RouteConvTranspose2D(
            features * 8, features * 4, name="upconv3", kernel_size=2, stride=2,
        )
        self.decoder3 = attention_UNet._block((features * 4) * 2, features * 4, name="dec3", bn_affine=bn_affine, bn_track=bn_track, prefix_name="decoder3.")
        self.upconv2 = RouteConvTranspose2D(
            features * 4, features * 2, name="upconv2", kernel_size=2, stride=2
        )
        self.decoder2 = attention_UNet._block((features * 2) * 2, features * 2, name="dec2", bn_affine=bn_affine, bn_track=bn_track, prefix_name="decoder2.")

        self.upconv1 = RouteConvTranspose2D(
            features * 2, features, kernel_size=2, name="upconv1", stride=2,
        )
        self.decoder1 = attention_UNet._block(features * 2, features, name="dec1", bn_affine=bn_affine, bn_track=bn_track, prefix_name="decoder1.")

        self.conv = RouteConv2D(
            in_channels=features, out_channels=out_channels, kernel_size=1, name="conv" 
        )
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(24)
        self.fc = nn.Linear(512, 4)
        self.flat = ViewFlatten()


    def forward(self, x, task):
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        if task == "self":
            logits = self.fc(self.flat(self.avgpool(self.relu(self.bn(bottleneck)))))
            return logits

        dec4 = self.upconv4(bottleneck)
        enc4 = self.att4(dec4, enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.att1(dec1, enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        dec1 = self.conv(dec1)
        

        return dec1

    @staticmethod
    def _block(in_channels, features, bn_affine, bn_track, name, prefix_name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        RouteConv2D(
                            in_channels=in_channels,
                            out_channels=features,
                            name = prefix_name + name + "_conv1",
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn1", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        RouteConv2D(
                            in_channels=features,
                            out_channels=features,
                            name = prefix_name + name + "_conv2",
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn2", nn.BatchNorm2d(num_features=features, affine=bn_affine, track_running_stats=bn_track)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
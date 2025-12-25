import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
from torchvision.models.resnet import ResNet50_Weights, ResNet, _ovewrite_named_param, Bottleneck, BasicBlock, conv1x1
from torchvision.utils import _log_api_usage_once

def resnet50_1(weights=None, progress: bool = True, **kwargs: Any):
    weights = ResNet50_Weights.verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = ResNet(newBottleneck, [3, 4, 6, 3], **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)
    return model

def resnet50_2(weights=None, progress: bool = True, **kwargs: Any):
    weights = ResNet50_Weights.verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = newResNet_2(Bottleneck, [3, 4, 6, 3], **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)
    return model

###########################################################################################################
###########################################################################################################

class newBottleneck(Bottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        width = int(planes * (base_width / 64.0)) * groups
        self.blk = blk(width)
        self.bn4 = nn.BatchNorm2d(width)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)     # 1x1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # 3x3
        out = self.bn2(out)
        out = self.relu(out)

        out = self.blk(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv3(out)   # 1x1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class newResnet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.blk_1 = blk(512)
        self.blk_2 = blk(1024)
        del self.avgpool
        del self.fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # print('ssss',x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        # print(x.shape)
        x = self.blk_1(x)
        x = self.layer3(x)

        # print(x.shape)
        x = self.blk_2(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x
    
class newResNet_2(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.blk_1 = blk(512)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.blk_2 = blk(1024)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.blk_1(x)
        x = self.layer3(x)
        x = self.blk_2(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
###########################################################################################################
###########################################################################################################

class blk(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        in_ch = in_ch // 4
        self.ce = ce(in_ch)
        self.lte = lte(in_ch)
        self.gte = gte(in_ch)
        self.se = se(in_ch)
        
    def forward(self, x):
        # (160, 512, 224, 224)
        n, c, h, w = x.shape
        # print(x.shape)
        # print(x[:,:c//4].shape, x[:, c//4:c//2].shape, x[:, c//2:c*3//4].shape, x[:, c*3//4:].shape)
        f1 = self.ce(x[:,:c//4])
        f2 = self.gte(x[:, c//4:c//2])
        f3 = self.lte(x[:, c//2:c*3//4])
        f4 = self.se(x[:, c*3//4:])
        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        x = torch.concat([f1, f2, f3, f4], dim=1)
        return x

class ce(nn.Module):
    def __init__(self, in_ch, seq_len=8):
        super().__init__()
        self.seq_len = seq_len

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm3d(in_ch)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        # (160, 64/g, 224, 224)
        n, c, h, w = x.shape
        x = x.reshape(-1, self.seq_len, c, h, w).transpose(1, 2)    # (20, 64/g, 8, 224, 224)
        res = x

        a, b = x[:, :, :-1], x[:, :, 1:]
        diff = b - a    # (20, 64/g, 7, 224, 224)
        diff = F.pad(diff, (0,0,0,0,0,1), 'constant', 0)    # (20, 64/g, 8, 224, 224)
        diff = self.avgpool(diff)   # (20, 64/g, 1, 1, 1)

        x = self.fc(diff)
        x = self.sigmoid(x) * res   # (20, 64/g, 8, 224, 224)
        return x.transpose(1, 2).reshape(n, c, h, w)

class gte(nn.Module):
    def __init__(self, in_ch, seq_len=8):
        super().__init__()
        self.seq_len = seq_len

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv3d(in_ch, in_ch*2, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.conv2 = nn.Conv3d(in_ch*2, in_ch, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.bn1 = nn.BatchNorm3d(in_ch*2)
        self.bn2 = nn.BatchNorm3d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        # (160, 64/g, 224, 224)
        n, c, h, w = x.shape
        # print(x.shape)
        x = x.reshape(-1, self.seq_len, c, h, w).transpose(1, 2)    # (20, 64/g, 8, 224, 224)
        res = x

        a, b = x[:, :, :-1], x[:, :, 1:]
        diff = b - a    # (20, 64/g, 7, 224, 224)
        diff = F.pad(diff, (0,0,0,0,0,1), 'constant', 0)    # (20, 64/g, 8, 224, 224)
        diff = self.avgpool(diff)   # (20, 64/g, 8, 1, 1)
        
        x = self.conv1(diff)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sigmoid(x) * res

        return x.transpose(1, 2).reshape(n, c, h, w)

class lte(nn.Module):
    def __init__(self, in_ch, seq_len=8):
        super().__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Conv3d(in_ch, in_ch*2, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.conv2 = nn.Conv3d(in_ch*2, in_ch, kernel_size=(3,1,1), stride=1, padding=(2,0,0), dilation=(2,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(in_ch*2)
        self.bn2 = nn.BatchNorm3d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        # (160, 64/g, 224, 224)
        n, c, h, w = x.shape
        # print(x.shape)
        x = x.reshape(-1, self.seq_len, c, h, w).transpose(1, 2)    # (20, 64/g, 8, 224, 224)
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sigmoid(x) * res 

        return x.transpose(1, 2).reshape(n, c, h, w)

class se(nn.Module):
    def __init__(self, in_ch, seq_len=8):
        super().__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Conv2d(in_ch, in_ch*2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv2 = nn.Conv2d(in_ch*2, in_ch, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch*2)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        # (160, 64/g, 224, 224)
        n, c, h, w = x.shape
        x = x.reshape(-1, self.seq_len, c, h, w).transpose(1, 2)    # (20, 64/g, 8, 224, 224)
        res = x

        x = x.mean(2)   # (20, 64/g, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sigmoid(x).unsqueeze(2) * res  # (20, 64/g, 8, 224, 224)
        return x.transpose(1, 2).reshape(n, c, h, w)

###########################################################################################################
###########################################################################################################

        
class GroupGLKA(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2*n_feats
        
        self.n_feats= n_feats
        self.i_feats = i_feats
        
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        #Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 7, 1, 7//2, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 9, stride=1, padding=(9//2)*4, groups=n_feats//3, dilation=4),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 7, stride=1, padding=(7//2)*3, groups=n_feats//3, dilation=3),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 5, stride=1, padding=(5//2)*2, groups=n_feats//3, dilation=2),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        
        self.X3 = nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3)
        self.X5 = nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3)
        self.X7 = nn.Conv2d(n_feats//3, n_feats//3, 7, 1, 7//2, groups= n_feats//3)
        
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        
    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()
        
        x = self.norm(x)
        
        x = self.proj_first(x)
        
        a, x = torch.chunk(x, 2, dim=1) 
        
        a_1, a_2, a_3= torch.chunk(a, 3, dim=1)
        
        a = torch.cat([self.LKA3(a_1)*self.X3(a_1), self.LKA5(a_2)*self.X5(a_2), self.LKA7(a_3)*self.X7(a_3)], dim=1)
        
        x = self.proj_last(x*a)*self.scale + shortcut
        
        return x  

###########################################################################################################
###########################################################################################################

from torch import einsum
from einops import rearrange


    
class mo_1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mo = nn.Parameter(torch.rand(1, 1, 2048), requires_grad=True)
        self.trans = Transformer_v1()

    def forward(self, qu, su, su_l):
        # (160, 2048) (200, 2048)
        qu_v = qu.reshape(-1, 8, 2048).mean(1).unsqueeze(0)  # (1, 20, 2048)
        su_v = su.reshape(-1, 8, 2048).mean(1).unsqueeze(0)  # (1, 25, 2048)
        mo_q = self.trans(qu_v, self.mo, self.mo).squeeze(0)   # (20, 2048)
        mo_s = self.trans(su_v, self.mo, self.mo).squeeze(0)   # (25, 2048)
        unique_labels = torch.unique(su_l)
        mo_s = [torch.mean(mo_s[extract_class_indices(su_l, c)], dim=0) for c in unique_labels] # 5 2048
        mo_s = torch.stack(mo_s, 0) # (5, 2048)

        dist = cos_sim(mo_q, mo_s)
        dist = 1 - dist
        # dist = cosine_dist(mo_q,mo_s) 
        #print(dist.shape)
        probability = torch.nn.functional.softmax(dist, dim=-1)
        return -probability.unsqueeze(0)

class mo_2(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mo = nn.Parameter(torch.rand(1, 1, 2048), requires_grad=True)
        self.trans_1 = Transformer_v1(dropout_atte=0.2)
        self.trans_2 = Transformer_v1(dropout_atte=0.2)

    def forward(self, qu, su, su_l):
        # (160, 2048) (200, 2048)
        qu = qu.reshape(-1, 8, 2048)    # (20, 8, 2048)
        su = su.reshape(-1, 8, 2048)    # (25, 8, 2048)

        qn, sn = qu.size(0), su.size(0)

        qu_v = qu.mean(1).unsqueeze(0)  # (1, 20, 2048)
        su_v = su.mean(1).unsqueeze(0)  # (1, 25, 2048)
        mo_q = self.trans_1(qu_v, self.mo, self.mo).reshape(qn, 1, 2048)   # (20, 1, 2048)
        mo_s = self.trans_1(su_v, self.mo, self.mo).reshape(sn, 1, 2048)   # (25, 1, 2048)

        qu_pre, qu_la = qu[:, :-1], qu[:, 1:]
        su_pre, su_la = su[:, :-1], su[:, 1:]
        diff_q = qu_la - qu_pre # (20, 7, 2048)
        diff_s = su_la - su_pre # (25, 7, 2048)
        
        mo_q = self.trans_2(mo_q, diff_q, diff_q).squeeze(1) # (20, 2048)
        mo_s = self.trans_2(mo_s, diff_s, diff_s).squeeze(1) # (25, 2048)

        dist = cosine_dist(mo_q,mo_s) # (20, 25)

        unique_labels = torch.unique(su_l)
        dist = [torch.mean(torch.index_select(dist, 1, extract_class_indices(su_l, c)), dim=1) for c in unique_labels] # 5 20
        dist = torch.stack(dist, 0).transpose(0, 1) # (5, 20)
        #print(dist.shape)
        probability = torch.nn.functional.softmax(dist, dim=-1)
        return probability.unsqueeze(0)

class mo_2_2(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mo = nn.Parameter(torch.rand(1, 1, 2048), requires_grad=True)
        self.trans_1 = Transformer_v1(dropout_atte=0.2)
        self.trans_2 = Transformer_v1(dropout_atte=0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, qu, su):
        # (160, 2048) (200, 2048)
        qu = qu.reshape(-1, 8, 2048)    # (20, 8, 2048)
        su = su.reshape(-1, 8, 2048)    # (25, 8, 2048)

        qn, sn = qu.size(0), su.size(0)

        qu_v = qu.mean(1).unsqueeze(0)  # (1, 20, 2048)
        su_v = su.mean(1).unsqueeze(0)  # (1, 25, 2048)
        mo_q = self.trans_1(qu_v, self.mo, self.mo).reshape(qn, 1, 2048)   # (20, 1, 2048)
        mo_s = self.trans_1(su_v, self.mo, self.mo).reshape(sn, 1, 2048)   # (25, 1, 2048)

        qu_pre, qu_la = qu[:, :-1], qu[:, 1:]
        su_pre, su_la = su[:, :-1], su[:, 1:]
        diff_q = qu_la - qu_pre # (20, 7, 2048)
        diff_s = su_la - su_pre # (25, 7, 2048)
        
        mo_q = self.relu(self.trans_2(mo_q, diff_q, diff_q)) # (20, 1, 2048)
        mo_s = self.relu(self.trans_2(mo_s, diff_s, diff_s)) # (25, 1, 2048)
        
        qu = qu * mo_q
        su = su * mo_s
        return qu.reshape(-1, 2048), su.reshape(-1, 2048)


class mo_3(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mo = nn.Parameter(torch.rand(1, 49, 2048), requires_grad=True)
        nn.init.xavier_normal_(self.mo)
        self.trans_1 = Transformer_v1(dropout_atte=0.2)
        self.trans_2 = Transformer_v1(dropout_atte=0.2)

    def forward(self, qu, su, su_l):
        # (160, 2048, 7, 7) (200, 2048, 7, 7)
        qu_v = qu.reshape(-1, 8, 2048, 49)  # (20, 8, 2048, 49)
        su_v = su.reshape(-1, 8, 2048, 49)  # (25, 8, 2048, 49)
        mo_q = self.mo * qu_v.mean((1,2,3), keepdim=True).squeeze(1)   # (20, 49, 2048)
        mo_s = self.mo * su_v.mean((1,2,3), keepdim=True).squeeze(1)   # (25, 49, 2048)
        for i in range(8):
            qu_v_f = qu_v[:, i].transpose(-2, -1)   # (20, 49, 2048)
            su_v_f = su_v[:, i].transpose(-2, -1)   # (25, 49, 2048)
            mo_q = self.trans_1(qu_v_f, mo_q, mo_q)   # (20, 49, 2048)
            mo_s = self.trans_1(su_v_f, mo_s, mo_s)   # (25, 49, 2048)
        mo_q = self.trans_2(mo_q, mo_q, mo_q).mean(1)   # (20, 2048)
        mo_s = self.trans_2(mo_s, mo_s, mo_s).mean(1)   # (25, 2048)

        dist = cosine_dist(mo_q,mo_s) # (20, 25)

        unique_labels = torch.unique(su_l)
        dist = [torch.mean(torch.index_select(dist, 1, extract_class_indices(su_l, c)), dim=1) for c in unique_labels] # 5 20
        dist = torch.stack(dist, 0).transpose(0, 1) # (5, 20)
        #print(dist.shape)
        probability = torch.nn.functional.softmax(dist, dim=-1)
        return probability.unsqueeze(0)
###########################################################################################################

import math
from torch.autograd import Variable
class PositionalEncoder(nn.Module):
    def __init__(self, d_model=2048, max_seq_len = 20, dropout = 0.1, A_scale=10., B_scale=1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.A_scale = A_scale
        self.B_scale = B_scale
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        
        x = x * math.sqrt(self.d_model/self.A_scale)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + self.B_scale * pe
        return self.dropout(x)
    
class PositionalEncoder_v2(nn.Module):
    def __init__(self, d_model=2048, max_seq_len = 20, dropout = 0.1, A_scale=10., B_scale=1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                if i + 1 == d_model:
                    continue
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.A_scale = A_scale
        self.B_scale = B_scale
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        
        x = x * math.sqrt(self.d_model/self.A_scale)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + self.B_scale * pe
        return self.dropout(x)

class t_group(nn.Module):
    def __init__(self,):
        super().__init__()
        # self.class_token = nn.Parameter(torch.randn(1, 1, 1, 2048))
        self.pe = PositionalEncoder(2048, dropout=0.1, A_scale=10., B_scale=1.)
        self.trans = Transformer_v1(dropout_atte=0.2, depth=2)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv1d(2048, 2048, 2, 1, 0, bias=False)
        # self.conv3 = nn.Conv1d(2048, 2048, 3, 1, 0, bias=False)
        self.conv2_1 = nn.Conv1d(2048, 2048, 1, 1, 0, bias=False)
        self.conv2_2 = nn.Conv1d(2048, 2048, 3, 1, 1, bias=False)
        self.conv2_3 = nn.Conv1d(2048, 2048, 2, 1, 0, dilation=2,  bias=False)

    def forward(self, qu, su, su_l):
        # (160, 2048) (200, 2048)
        qu_v = qu.reshape(-1, 8, 2048)  # (20, 8, 2048)
        su_v = su.reshape(-1, 8, 2048)  # (25, 8, 2048)
        qu_v = F.pad(qu_v, (0,0,1,1), 'replicate')  # (20, 10, 2048)
        su_v = F.pad(su_v, (0,0,1,1), 'replicate')  # (25, 10, 2048)

        qu_v = self.pe(qu_v)
        su_v = self.pe(su_v)

        qu_v_f = [qu_v[:, 2*i:2*i+4] for i in range(4)] # 4 (20, 4, 2048)
        su_v_f = [su_v[:, 2*i:2*i+4] for i in range(4)] # 4 (25, 4, 2048)

        qu_v_f = torch.stack(qu_v_f, dim=1).reshape(-1, 4, 2048) # (20*4, 4, 2048)
        su_v_f = torch.stack(su_v_f, dim=1).reshape(-1, 4, 2048) # (25*4, 4, 2048)

        # qu_v_f = torch.cat((self.class_token.expand(qu_v_f.size(0), 4, -1, -1), qu_v_f), dim=2) # (20, 4, 5, 2048)
        # su_v_f = torch.cat((self.class_token.expand(su_v_f.size(0), 4, -1, -1), su_v_f), dim=2) # (25, 4, 5, 2048)

        q_p_l_2 = self.relu(self.conv2_3(self.relu(self.conv2_2(self.relu(self.conv2_1(qu_v_f.transpose(-2, -1))))))).mean(-1)  # (20*4, 2048, 3) -> (20*4, 2048)
        s_p_l_2 = self.relu(self.conv2_3(self.relu(self.conv2_2(self.relu(self.conv2_1(su_v_f.transpose(-2, -1))))))).mean(-1)  # (25*4, 2048, 3) -> (25*4, 2048)

        print(q_p_l_2.shape)
        # q_p_l_3 = self.relu(self.conv3(qu_v_f.transpose(-2, -1))).mean(-1)  # (20*4, 2048, 2) -> (20*4, 2048)
        # s_p_l_3 = self.relu(self.conv2(su_v_f.transpose(-2, -1))).mean(-1)  # (25*4, 2048, 2) -> (25*4, 2048)

        q_p_g = self.relu(self.trans(qu_v_f, qu_v_f, qu_v_f)).mean(-2) # (20*4, 2048)
        s_p_g = self.relu(self.trans(su_v_f, su_v_f, su_v_f)).mean(-2) # (25*4, 2048)

        q_p = qu_v_f.mean(1) + q_p_l_2  + q_p_g
        s_p = su_v_f.mean(1) + s_p_l_2  + s_p_g

        dist = cos_sim(q_p, s_p) # (20*4, 25*4)
        dist = 1 - dist
        # dist = cosine_dist(q_p, s_p) # (20*4, 25*4)
        dist = rearrange(dist, '(tb ts) (sb ss) -> tb sb ts ss', tb = qu_v.size(0), sb = su_v.size(0))  # [20, 25, 4, 4]
        dist = dist.diagonal(dim1=-2, dim2=-1).sum(-1)  #(20, 25)
        # dist = dist.diagonal(dim1=-2, dim2=-1)  #(20, 25, 4)
        # mask = torch.ones_like(dist)
        # mask[:,:, [0,3]] = 0.1 
        # dist = (dist * mask).mean(-1)

        unique_labels = torch.unique(su_l)
        class_dists = [torch.mean(torch.index_select(dist, 1, extract_class_indices(su_l, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        # class_dists = class_dists.softmax(-1)

        return -class_dists.unsqueeze(0)

###########################################################################################################

def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

def cosine_dist(x,y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    cosine_sim_list = []
    for i in range(m):
        y_tmp = y[i].unsqueeze(0)
        x_tmp = x
        #print(x_tmp.size(),y_tmp.size())
        cosine_sim = nn.functional.cosine_similarity(x_tmp,y_tmp)
        cosine_sim_list.append(cosine_sim)
    return torch.stack(cosine_sim_list).transpose(0,1)

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

def euclidean_dist(x, y, timewise=False):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    # return: N x M
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]

def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]

###########################################################################################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention_v3(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b  i d, b  j d -> b  i j', q, k) * self.scale

        attn = self.attend(dots)  # [30, 8, 8, 5]

        out = einsum('b  i j, b  j d -> b  i d', attn, v)

        return self.to_out(out)
    
class Attention_qkv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)  # [30, 8, 8, 5]

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x
    
class PreNormattention_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs) + q

class PostNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)
           
class Transformer_v1(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte=0.05, mlp_dim=2048,
                 dropout_ffn=0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                    # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                    PreNormattention_qkv(dim,
                                         Attention_qkv(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, q, k, v):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(q, k, v)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x, x, x)
                x = ff(x) + x
        return x
    
class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte=0.05, mlp_dim=2048,
                 dropout_ffn=0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                    # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                    PreNormattention(dim, Attention(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x
    
###########################################################################################################
###########################################################################################################
if __name__ == '__main__':
    import torchvision.models as models
    from time import time
    # mm = resnet50_2(weights=models.ResNet50_Weights.DEFAULT)
    # mm = nn.Sequential(*list(mm.children())[:-1])
    # print(mm)
    mm = t_group()
    i = torch.ones(2, 8, 2048)
    i2 = torch.ones(40, 2048)
    pe = PositionalEncoder(2048,8)
    print(pe.training)
    print(pe(i).shape)
    start = time()
    # o1 = mm(i,i2, torch.tensor([0,2,1,3,4]))
    # loss = -F.cross_entropy(o1, torch.tensor([0]), reduction='none')
    # print(time()-start)
    # print(o1,loss)

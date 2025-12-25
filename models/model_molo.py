import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import math

from torch.autograd import Variable
import torchvision.models as models
from einops import rearrange
from torch.autograd import Variable

import os

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


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector



class CNN_FSHead(nn.Module):
    """
    Base class which handles a few-shot method. Contains a resnet backbone which computes features.
    """
    def __init__(self, cfg):
        super(CNN_FSHead, self).__init__()
        args = cfg
        self.train()
        self.args = args

        last_layer_idx = -1
        
        if self.args.MODEL.BACKBONE == "resnet18":
            backbone = models.resnet18(pretrained=True) 
            self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

        elif self.args.MODEL.BACKBONE == "resnet34":
            backbone = models.resnet34(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

        elif self.args.MODEL.BACKBONE == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()
        target_features = self.backbone(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)
        target_features = target_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)

        return support_features, target_features

    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can also contain
        other info needed to compute the loss. E.g. inter class distances.
        """
        raise NotImplementedError

    def distribute_model(self):
        """
        Use to split the backbone evenly over all GPUs. Modify if you have other components
        """
        if self.args.DEVICE.NUM_GPUS > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.DEVICE.NUM_GPUS)])
    
    def loss(self, task_dict, model_dict):
        """
        Takes in a the task dict containing labels etc.
        Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
        Default is cross entropy loss.
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention(dim, Attention(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
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

class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=kernel_size, groups=1)
            self.conv = DoubleConv2(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CNN_BiMHM_MoLo(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_BiMHM_MoLo, self).__init__(cfg)
        args = cfg
        self.args = cfg
        last_layer_idx = -1
        self.backbone = nn.Sequential(*list(self.backbone.children())[:last_layer_idx])
        if hasattr(self.args.MODEL,"USE_CONTRASTIVE") and self.args.MODEL.USE_CONTRASTIVE:
            if hasattr(self.args.MODEL,"TEMP_COFF") and self.args.MODEL.TEMP_COFF:
                self.scale = self.args.MODEL.TEMP_COFF
                self.scale_motion = self.args.MODEL.TEMP_COFF
            else:
                self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.scale.data.fill_(1.0)

                self.scale_motion = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.scale_motion.data.fill_(1.0)

        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        if self.args.MODEL.BACKBONE == "resnet50":
            self.mid_dim = 2048
            # self.mid_dim = 256
            self.pre_reduce = nn.Sequential()
            # self.pre_reduce = nn.Conv2d(2048, 256, kernel_size=1, padding=0, groups=4)
        else:
            self.mid_dim = 512
            self.pre_reduce = nn.Sequential()
            # self.pre_reduce = nn.Conv2d(512, 512, kernel_size=1, padding=0)  # nn.Sequential()
        if hasattr(self.args.MODEL,"POSITION_A") and hasattr(self.args.MODEL,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.MODEL.POSITION_A, B_scale=self.args.MODEL.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        self.class_token_motion = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        if hasattr(self.args.MODEL,"HEAD") and self.args.MODEL.HEAD:
            self.temporal_atte_before = Transformer_v2(dim=self.mid_dim, heads = self.args.MODEL.HEAD, dim_head_k = self.mid_dim//self.args.MODEL.HEAD, dropout_atte = 0.2)
            self.temporal_atte_before_motion = Transformer_v2(dim=self.mid_dim, heads = self.args.MODEL.HEAD, dim_head_k = self.mid_dim//self.args.MODEL.HEAD, dropout_atte = 0.2)
            
        else:
            
            self.temporal_atte_before = Transformer_v2(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
            self.temporal_atte_before_motion = Transformer_v2(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.factor = 8
        self.motion_reduce = nn.Conv3d(self.mid_dim, self.mid_dim//self.factor, kernel_size=(3,3,3), padding=(1,1,1), groups=1)
        self.motion_conv = nn.Conv2d(self.mid_dim//self.factor, self.mid_dim//self.factor, kernel_size=3, padding=1, groups=1)
        self.motion_up = nn.Conv2d(self.mid_dim//self.factor, self.mid_dim, kernel_size=1, padding=0, groups=1)
        if hasattr(self.args.MODEL, "USE_CLASSIFICATION") and self.args.MODEL.USE_CLASSIFICATION:
            if hasattr(self.args.DATA, "NUM_CLASS"):
                self.classification_layer = nn.Linear(self.mid_dim, int(self.args.DATA.NUM_CLASS))
            else:
                self.classification_layer = nn.Linear(self.mid_dim, 64)
        
        bilinear = True
        # factor = 2 if bilinear else 1
        factor = 1
        n_classes = 3
        self.up1 = Up2(self.mid_dim//self.factor, 128 // factor, bilinear, kernel_size=2)
        self.up2 = Up2(128, 32 // factor, bilinear, kernel_size=4)
        self.up3 = Up2(32, 16, bilinear, kernel_size=4)
        self.outc = OutConv(16, n_classes)
        # set_trace()
        
    

    def get_feats(self, support_images, target_images, support_labels):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.pre_reduce(self.backbone(support_images)).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.pre_reduce(self.backbone(target_images)).squeeze()   # [200, 2048, 7, 7]
        #print(support_features[0], support_labels)
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        support_features_motion = self.motion_reduce(support_features.reshape(-1, self.args.DATA.SEQ_LEN, dim, 7, 7).permute(0,2,1,3,4)).permute(0,2,1,3,4).reshape(-1, dim//self.factor, 7, 7)   # [40, 128, 7, 7]
        target_features_motion = self.motion_reduce(target_features.reshape(-1, self.args.DATA.SEQ_LEN, dim, 7, 7).permute(0,2,1,3,4)).permute(0,2,1,3,4).reshape(-1, dim//self.factor, 7, 7)
        support_features_motion_conv = self.motion_conv(support_features_motion)   # [40, 128, 7, 7]
        target_features_motion_conv = self.motion_conv(target_features_motion)
        support_features_motion = support_features_motion_conv.reshape(-1, self.args.DATA.SEQ_LEN, dim//self.factor, 7, 7)[:,1:] - support_features_motion.reshape(-1, self.args.DATA.SEQ_LEN, dim//self.factor, 7, 7)[:,:-1]
        support_features_motion = support_features_motion.reshape(-1, dim//self.factor, 7, 7)
        # support_features_motion = self.relu(self.motion_up(support_features_motion))

        target_features_motion = target_features_motion_conv.reshape(-1, self.args.DATA.SEQ_LEN, dim//self.factor, 7, 7)[:,1:] - target_features_motion.reshape(-1, self.args.DATA.SEQ_LEN, dim//self.factor, 7, 7)[:,:-1]
        target_features_motion = target_features_motion.reshape(-1, dim//self.factor, 7, 7)
        # target_features_motion = self.relu(self.motion_up(target_features_motion))
        feature_motion_recons = torch.cat([support_features_motion, target_features_motion], dim=0)
        feature_motion_recons = self.up1(feature_motion_recons)
        feature_motion_recons = self.up2(feature_motion_recons)
        feature_motion_recons = self.up3(feature_motion_recons)
        
        feature_motion_recons = self.outc(feature_motion_recons)
        support_features_motion = self.relu(self.motion_up(support_features_motion))
        target_features_motion = self.relu(self.motion_up(target_features_motion))
        support_features_motion = self.avg_pool(support_features_motion).squeeze().reshape(-1, self.args.DATA.SEQ_LEN-1, dim)
        target_features_motion = self.avg_pool(target_features_motion).squeeze().reshape(-1, self.args.DATA.SEQ_LEN-1, dim)
        support_bs = int(support_features_motion.shape[0])
        target_bs = int(target_features_motion.shape[0])
        support_features_motion = torch.cat((self.class_token_motion.expand(support_bs, -1, -1), support_features_motion), dim=1)
        target_features_motion = torch.cat((self.class_token_motion.expand(target_bs, -1, -1), target_features_motion), dim=1)
        target_features_motion = self.relu(self.temporal_atte_before_motion(self.pe(target_features_motion)))   # [5, 9, 2048]
        support_features_motion = self.relu(self.temporal_atte_before_motion(self.pe(support_features_motion)))

        


        support_features = self.avg_pool(support_features).squeeze()
        target_features = self.avg_pool(target_features).squeeze()

        Query_num = target_features.shape[0]//self.args.DATA.SEQ_LEN
        support_features = support_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)
        target_features = target_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)
        # support_features = self.temporal_atte(support_features.reshape(-1, self.args.DATA.SEQ_LEN, dim), support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)).unsqueeze(0).repeat(Query_num,1,1,1)   # [35, 5, 8, 2048]   V0
        # target_features = self.temporal_atte(target_features.reshape(-1, self.args.DATA.SEQ_LEN, dim), target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))# .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]
        support_bs = int(support_features.shape[0])
        target_bs = int(target_features.shape[0])
        support_features = torch.cat((self.class_token.expand(support_bs, -1, -1), support_features), dim=1)
        target_features = torch.cat((self.class_token.expand(target_bs, -1, -1), target_features), dim=1)
        support_features = self.relu(self.temporal_atte_before(self.pe(support_features)))   # [5, 9, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features)))   # .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]

        if hasattr(self.args.MODEL, "USE_CLASSIFICATION") and self.args.MODEL.USE_CLASSIFICATION:
            if hasattr(self.args.DATA, "NUM_CLASS"):
                if hasattr(self.args.MODEL, "USE_LOCAL") and self.args.MODEL.USE_LOCAL:
                    class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.DATA.NUM_CLASS))
                else:
                    class_logits = self.classification_layer(torch.cat([support_features.mean(1)+support_features_motion.mean(1), target_features.mean(1)+target_features_motion.mean(1)], 0))
            else:
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        else:
            class_logits = None
        

        unique_labels = torch.unique(support_labels)
        support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_features_motion = [torch.mean(torch.index_select(support_features_motion, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features_motion = torch.stack(support_features_motion)
        
        return support_features, target_features, class_logits, support_features_motion, target_features_motion, feature_motion_recons

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['context_images'], inputs['context_labels'], inputs['target_images']
        #support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        # [200, 3, 84, 84]
        # if self.training and hasattr(self.args.MODEL, "USE_FLOW"):
        #     support_images_re = inputs["support_set_flow"].reshape(-1, self.args.DATA.SEQ_LEN,3, 224, 224)[:,:(self.args.DATA.SEQ_LEN-1),:,:,:]
        #     target_images_re = inputs["target_set_flow"].reshape(-1, self.args.DATA.SEQ_LEN,3, 224, 224)[:,:(self.args.DATA.SEQ_LEN-1),:,:,:]
        #     input_recons = torch.cat([support_images_re, target_images_re], dim=0).reshape(-1, 3, 224, 224)
        # else:
            # support_images_re = support_images.reshape(-1, self.args.DATA.SEQ_LEN,3, 224, 224)
            # target_images_re = target_images.reshape(-1, self.args.DATA.SEQ_LEN,3, 224, 224)
            # # support_images, support_labels, target_images = inputs
            # input_recons = torch.cat([support_images_re[:,1:,:]-support_images_re[:,:-1,:], target_images_re[:,1:,:]- target_images_re[:,:-1,:]], dim=0).reshape(-1, 3, 224, 224)
        support_images_re = support_images.reshape(-1, self.args.DATA.SEQ_LEN,3, 224, 224)
        target_images_re = target_images.reshape(-1, self.args.DATA.SEQ_LEN,3, 224, 224)
        # support_images, support_labels, target_images = inputs
        input_recons = torch.cat([support_images_re[:,1:,:]-support_images_re[:,:-1,:], target_images_re[:,1:,:]- target_images_re[:,:-1,:]], dim=0).reshape(-1, 3, 224, 224)

        support_features, target_features, class_logits, support_features_motion, target_features_motion, feature_motion_recons = self.get_feats(support_images, target_images, support_labels)
        
        # 
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        # global
        support_features_g = support_features[:,0,:]
        target_features_g = target_features[:,0,:]
        support_features = support_features[:,1:,:]
        target_features = target_features[:,1:,:]

        # support to query
        class_sim_s2q = cos_sim(support_features, target_features_g)  # [5, 8, 35]
        class_dists_s2q = 1 - class_sim_s2q
        class_dists_s2q = [torch.sum(torch.index_select(class_dists_s2q, 0, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_s2q = torch.stack(class_dists_s2q).squeeze(1)
        if hasattr(self.args.MODEL,"USE_CONTRASTIVE") and self.args.MODEL.USE_CONTRASTIVE:
            class_dists_s2q = rearrange(class_dists_s2q * self.scale, 'c q -> q c')

        # query to support 
        class_sim_q2s = cos_sim(target_features, support_features_g)  # [35, 8, 5]
        class_dists_q2s = 1 - class_sim_q2s   
        class_dists_q2s = [torch.sum(torch.index_select(class_dists_q2s, 2, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_q2s = torch.stack(class_dists_q2s).squeeze(2)
        if hasattr(self.args.MODEL,"USE_CONTRASTIVE") and self.args.MODEL.USE_CONTRASTIVE:
            class_dists_q2s = rearrange(class_dists_q2s * self.scale, 'c q -> q c')

        # global
        support_features_motion_g = support_features_motion[:,0,:]
        target_features_motion_g = target_features_motion[:,0,:]
        support_features_motion = support_features_motion[:,1:,:]
        target_features_motion = target_features_motion[:,1:,:]

        # support to query
        class_sim_s2q_motion = cos_sim(support_features_motion, target_features_motion_g)  # [5, 8, 35]
        class_dists_s2q_motion = 1 - class_sim_s2q_motion
        class_dists_s2q_motion = [torch.sum(torch.index_select(class_dists_s2q_motion, 0, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_s2q_motion = torch.stack(class_dists_s2q_motion).squeeze(1)
        if hasattr(self.args.MODEL,"USE_CONTRASTIVE") and self.args.MODEL.USE_CONTRASTIVE:
            class_dists_s2q_motion = rearrange(class_dists_s2q_motion * self.scale_motion, 'c q -> q c')

        # query to support 
        class_sim_q2s_motion = cos_sim(target_features_motion, support_features_motion_g)  # [35, 8, 5]
        class_dists_q2s_motion = 1 - class_sim_q2s_motion   
        class_dists_q2s_motion = [torch.sum(torch.index_select(class_dists_q2s_motion, 2, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_q2s_motion = torch.stack(class_dists_q2s_motion).squeeze(2)
        if hasattr(self.args.MODEL,"USE_CONTRASTIVE") and self.args.MODEL.USE_CONTRASTIVE:
            class_dists_q2s_motion = rearrange(class_dists_q2s_motion * self.scale_motion, 'c q -> q c')

        support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
        target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]

        frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_dists = 1 - frame_sim
        
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

        # calculate query -> support and support -> query
        if hasattr(self.args.MODEL, "SINGLE_DIRECT") and self.args.MODEL.SINGLE_DIRECT:
            cum_dists = dists.min(3)[0].sum(2)
        else:
            cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)
            

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')

        support_features_motion = rearrange(support_features_motion, 'b s d -> (b s) d')  # [200, 2048]
        target_features_motion = rearrange(target_features_motion, 'b s d -> (b s) d')    # [200, 2048]
        frame_sim_motion = cos_sim(target_features_motion, support_features_motion)    # [200, 200]
        frame_dists_motion = 1 - frame_sim_motion   
        dists_motion = rearrange(frame_dists_motion, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]
        # calculate query -> support and support -> query
        if hasattr(self.args.MODEL, "SINGLE_DIRECT") and self.args.MODEL.SINGLE_DIRECT:
            # cum_dists_motion = OTAM_cum_dist(dists_motion)
            cum_dists_motion = dists_motion.min(3)[0].sum(2)
        else:
            cum_dists_motion = dists_motion.min(3)[0].sum(2) + dists_motion.min(2)[0].sum(2)
        class_dists_motion = [torch.mean(torch.index_select(cum_dists_motion, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_motion = torch.stack(class_dists_motion)
        class_dists_motion = rearrange(class_dists_motion, 'c q -> q c')
        
        if hasattr(self.args.MODEL, "LOGIT_BALANCE_COFF") and self.args.MODEL.LOGIT_BALANCE_COFF:
            class_dists = class_dists + self.args.MODEL.LOGIT_BALANCE_COFF*class_dists_motion
        else:
            class_dists = class_dists + 0.3*class_dists_motion
        
        if self.training:
            loss_recons = (feature_motion_recons - input_recons) ** 2   # [280, 3, 224, 224]
            loss_recons = loss_recons.mean()  # [N, L], mean loss per patch
        else:
            loss_recons = torch.tensor(0.)
        return_dict = {'logits': - class_dists.unsqueeze(0) , 'class_logits': class_logits.unsqueeze(0), "logits_s2q": -class_dists_s2q.unsqueeze(0), "logits_q2s": -class_dists_q2s.unsqueeze(0), "logits_s2q_motion": -class_dists_s2q_motion.unsqueeze(0), "logits_q2s_motion": -class_dists_q2s_motion.unsqueeze(0), "loss_recons": loss_recons,}
        
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

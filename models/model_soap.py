import torch
import torch.nn as nn
from collections import OrderedDict
# from utils.utils import split_first_dim_linear
import math
import numpy as np
from itertools import combinations 

from torch.autograd import Variable

import torchvision.models as models

NUM_SAMPLES=1

def cls_d(x):
    # (5, 8, 2048)
    prototypes = x.mean(1)
    diff = prototypes.unsqueeze(1) - prototypes.unsqueeze(0)
    # 计算平方和
    square_diff = torch.sum(diff ** 2, dim=2)  # (5, 5)
    # 防止数值不稳定，确保没有负数或零
    square_diff = torch.clamp(square_diff, min=1e-12)
    distances = torch.sqrt(square_diff)
    # print(distances)
    mask = torch.ones_like(distances)
    torch.diagonal(mask)[:] = 0
    distances = distances * mask
    lmd = nn.Parameter(torch.rand(1), requires_grad=True)
    loss = torch.exp(-lmd * torch.mean(distances)**2)
    return loss

def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)



class TemporalCrossTransformer(nn.Module):
    def __init__(self, cfg, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.cfg = cfg
        self.temporal_set_size = temporal_set_size

        max_len = int(self.cfg.DATA.SEQ_LEN * 1.5)
        self.pe = PositionalEncoding(self.cfg.trans_linear_in_dim, self.cfg.MODEL.TRANS_DROPOUT, max_len=max_len)

        self.k_linear = nn.Linear(self.cfg.trans_linear_in_dim*temporal_set_size, self.cfg.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.cfg.trans_linear_in_dim*temporal_set_size, self.cfg.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.cfg.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.cfg.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.cfg.DATA.SEQ_LEN)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.cfg.TRAIN.WAY).cuda()

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.cfg.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict



    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

class HMEM(nn.Module):
    def __init__(self, T):
        super(HMEM, self).__init__()
        self.T = T
        self.conv = nn.Conv2d(3, 3, 3, 1, 1, device='cuda')


    def forward(self, x):
        N, T, C, H, W = x.shape
        conv_f = self.conv(x.reshape(-1, C, H, W)).reshape(N, T, C, H, W)
        laster = None
        res = []
        idx = [i for i in range(T - self.T)]
        idx.reverse()
        for i in idx:
            laster = conv_f[:, i:i+self.T, :, :, :]
            if i == T - self.T:
                continue
            res.append(laster - x[:, i:i+self.T, :, :, :])  # (N, self.T, C, H, W)
        res = torch.concat(res, dim=1)  # (N, self.T*(T-self.T+1), C, H, W)
        return res

class SOAP(nn.Module):
    def __init__(self, cfg):
        super(SOAP, self).__init__()

        self.cr = cfg.MODEL.cr
        f_cnt = 0
        self.hmem_ls = []

        for T in cfg.MODEL.O:
            f_cnt += T * (cfg.DATA.SEQ_LEN - T)
            self.hmem_ls.append(HMEM(T))
        self.f_cnt = f_cnt
        self.conv_st = nn.Conv3d(1, 1, 3, 1, 1, bias=False)
        self.conv_ch1 = nn.Conv2d(3, self.cr, 1, bias=False)
        self.conv_ch2 = nn.Conv2d(self.cr, 3, 1, bias=False)

        self.conv_ch_1d = nn.Conv2d(self.cr, self.cr, (3, 1), 1, (1, 0))
        self.hm_lin = nn.Linear(f_cnt, cfg.DATA.SEQ_LEN)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()



    def forward(self, su, qu):
        # su: (25, 8, 3, 224, 224) | qu: (20, 8, 3, 224, 224)
        sn, T, C, H, W = su.shape
        qn = qu.size(0)
        # 3DEM
        f_s = su.mean(2, keepdim=True).transpose(1, 2)  # (25, 1, 8, 224, 224)
        f_q = qu.mean(2, keepdim=True).transpose(1, 2)  # (20, 1, 8, 224, 224)
        f_s = self.conv_st(f_s)
        f_q = self.conv_st(f_q)
        tdem_s = self.sig(f_s.transpose(1, 2)) * su + su  # (25, 8, 3, 224, 224)
        tdem_q = self.sig(f_q.transpose(1, 2)) * qu + qu  # (20, 8, 3, 224, 224)

        # CWEM
        f_s = self.avgpool(su).reshape(-1, C, 1, 1)  # (200, 3, 1, 1)
        f_q = self.avgpool(qu).reshape(-1, C, 1, 1)  # (160, 3, 1, 1)
        f_s = self.conv_ch1(f_s).reshape(sn, T, self.cr, 1, 1).squeeze(-1).transpose(1, 2)  # (25, 16, 8, 1)
        f_q = self.conv_ch1(f_q).reshape(qn, T, self.cr, 1, 1).squeeze(-1).transpose(1, 2)  # (20, 16, 8, 1)
        f_s = self.conv_ch_1d(f_s).transpose(1, 2).unsqueeze(-1)  # (25, 8, 16, 1, 1)
        f_q = self.conv_ch_1d(f_q).transpose(1, 2).unsqueeze(-1)  # (20, 8, 16, 1, 1)
        f_s = self.conv_ch2(f_s.reshape(-1, self.cr, 1, 1)).reshape(sn, T, C, 1, 1)    # (25, 8, 3, 1, 1)
        f_q = self.conv_ch2(f_q.reshape(-1, self.cr, 1, 1)).reshape(qn, T, C, 1, 1)    # (20, 8, 3, 1, 1)
        cwem_s = self.sig(f_s) * su + su
        cwem_q = self.sig(f_q) * qu + qu

        # HMEM
        f_s = torch.concat([i(su) for i in self.hmem_ls], dim=1)    # (25, x, 3, 224, 224)
        f_q = torch.concat([i(qu) for i in self.hmem_ls], dim=1)    # (20, x, 3, 224, 224)
        f_s = self.hm_lin(f_s.reshape(sn, self.f_cnt, -1).transpose(-1, -2)).transpose(-1, -2).reshape(sn, T, C, H, W)
        f_q = self.hm_lin(f_q.reshape(qn, self.f_cnt, -1).transpose(-1, -2)).transpose(-1, -2).reshape(qn, T, C, H, W)
        hmem_s = self.sig(self.avgpool(f_s)) * su + su
        hmem_q = self.sig(self.avgpool(f_q)) * qu + qu

        su = su + tdem_s + cwem_s + hmem_s
        qu = qu + tdem_q + cwem_q + hmem_q
        return su, qu




class CNN_SOAP(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, cfg):
        super(CNN_SOAP, self).__init__()

        self.train()
        self.cfg = cfg

        if self.cfg.MODEL.BACKBONE == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif self.cfg.MODEL.BACKBONE == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif self.cfg.MODEL.BACKBONE == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        last_layer_idx = -1
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])

        self.tripel_prior = SOAP(cfg)
        self.transformers = nn.ModuleList([TemporalCrossTransformer(cfg, s) for s in cfg.MODEL.TEMP_SET]) 

    def forward(self, inputs):

        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''
        context_images, context_labels, target_images = inputs['context_images'], inputs['context_labels'], inputs['target_images']
        #print(context_images.shape,target_images.shape)
        _, C, H, W = context_images.shape
        su, qu = self.tripel_prior(context_images.reshape(-1, self.cfg.DATA.SEQ_LEN, C, H, W), target_images.reshape(-1, self.cfg.DATA.SEQ_LEN, C, H, W))
        context_features = self.resnet(su.reshape(-1, C, H, W)).squeeze() # 200 x 2048 
        target_features = self.resnet(qu.reshape(-1, C, H, W)).squeeze() # 160 x 2048

        dim = int(context_features.shape[1])

        t_loss = cls_d(context_features.reshape(-1, self.cfg.DATA.SEQ_LEN, dim))

        context_features = context_features.reshape(-1, self.cfg.DATA.SEQ_LEN, dim)
        target_features = target_features.reshape(-1, self.cfg.DATA.SEQ_LEN, dim)

        all_logits = [t(context_features, context_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.shape[0]]), 't_loss':t_loss}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.cfg.DEVICE.NUM_GPUS > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.cfg.DEVICE.NUM_GPUS)])

            self.transformers.cuda(0)
            self.new_dist_loss_post_pat = [n.cuda(0) for n in self.new_dist_loss_post_pat]

            self.attn_pat.cuda(0)
            self.attn_pat = torch.nn.DataParallel(self.attn_pat, device_ids=[i for i in range(0, self.cfg.DEVICE.NUM_GPUS)])

            self.fr_enrich.cuda(0)
            self.fr_enrich = torch.nn.DataParallel(self.fr_enrich, device_ids=[i for i in range(0, self.cfg.DEVICE.NUM_GPUS)])

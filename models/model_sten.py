import os
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models
from .clip_fsar import load,tokenize
from .myRes import Transformer_v1, cos_sim, extract_class_indices, Attention_qkv, PositionalEncoder

class CNN_OTAM_CLIPFSAR(nn.Module):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_OTAM_CLIPFSAR, self).__init__()
        self.args = cfg
        if cfg.MODEL.BACKBONE=="RN50":
            backbone, self.preprocess = load(cfg.MODEL.BACKBONE, device="cuda", cfg=cfg, jit=False)   # ViT-B/16
            self.backbone = backbone.visual    # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 1024
            
            #for name, param in backbone.named_parameters():
            #    if 'layer4' in name:
            #        break
            #    #print(name)
            #    param.requires_grad = False
        elif cfg.MODEL.BACKBONE=="ViT-B/16":
            backbone, self.preprocess = load(cfg.MODEL.BACKBONE, device="cuda", cfg=cfg, jit=False)   # ViT-B/16
            self.backbone = backbone.visual   # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            # backbone, self.preprocess = load("RN50", device="cuda", cfg=cfg, jit=False)
            # self.backbone = backbone.visual model.load_state_dict(state_dict)
            # self.backbone = CLIP
            self.mid_dim = 512
        with torch.no_grad():
            text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
            text_templete = tokenize(text_templete).cuda()
            self.text_features_train = backbone.encode_text(text_templete)

            text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
            text_templete = tokenize(text_templete).cuda()
            self.text_features_test = backbone.encode_text(text_templete)
        
        
        #self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #self.scale.data.fill_(1.0)
        
        #self.cmf = Attention_qkv(dim=self.mid_dim, heads=4, dim_head=512, dropout=0.3)
        # self.cmf = Transformer_v1(dim=self.mid_dim, heads = 4, dim_head_k = 512, dropout_atte = 0.3, depth=1)
        #self.norm = nn.LayerNorm(self.mid_dim)
        #self.mapm = Transformer_v1(dim=self.mid_dim, heads = 4, dim_head_k = 512, dropout_atte = 0.3, depth=1)

        #self.tem = Transformer_v1(dim=self.mid_dim, heads = 4, dim_head_k = 512, dropout_atte = 0.3, depth=2)
        #self.pe = PositionalEncoder(d_model=self.mid_dim)
        
        # set_trace()
    
    def forward(self, inputs):
        support_images, support_labels, target_images, support_real_class = inputs['context_images'], inputs['context_labels'], inputs['target_images'], inputs['real_support_labels'] # [200, 3, 224, 224] inputs["real_support_labels"]
        
        su_f = self.backbone(support_images).reshape(-1, 8, self.mid_dim).mean(1)    # (25, 8, 1024)
        qu_f = self.backbone(target_images).reshape(-1, 8, self.mid_dim).mean(1)     # (20, 8, 1024)

        if self.training:
            t_f = self.text_features_train[support_real_class.long()]  # (25, 1, 1024)
        else:
            t_f = self.text_features_test[support_real_class.long()]   # (25, 1, 1024)
        #su_f = self.cmf(t_f, su_f, su_f) + su_f
        #su_f = self.norm(su_f)

        #qu_f = self.mapm(qu_f, qu_f, qu_f)

        #su_f = self.pe(su_f)    # (5, 8, 1024)
        #qu_f = self.pe(qu_f)    # (20, 8, 1024)

        #su_p = self.tem(su_f, su_f, su_f)
        #qu_p = self.tem(qu_f, qu_f, qu_f)

        #dist = cos_sim(qu_p.reshape(-1, self.mid_dim), su_p.reshape(-1, self.mid_dim)) # (20*4, 25*4)
        #dist = 1 - dist
        #dist = rearrange(dist, '(tb ts) (sb ss) -> tb sb ts ss', tb = qu_f.size(0), sb = su_f.size(0))  # [20, 25, 4, 4]
        
        #unique_labels = torch.unique(support_labels)
        #class_dists = [torch.mean(torch.index_select(dist.mean((-2,-1)), 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        #class_dists = torch.stack(class_dists)
        #class_dists = rearrange(class_dists * self.scale, 'c q -> q c')

        unique_labels = torch.unique(support_labels)
        t_f = [torch.mean(torch.index_select(t_f, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        t_f = torch.stack(t_f) # (5, 1024)

        su_f = [torch.mean(torch.index_select(su_f, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        su_f = torch.stack(su_f)

        sim_qt = cos_sim(qu_f, t_f).softmax(-1)  # (20, 5)
        sim_qs = cos_sim(qu_f, su_f).softmax(-1)  # (20, 5)
        sim = sim_qt * sim_qs
        
        return {'logits': sim.unsqueeze(0)}











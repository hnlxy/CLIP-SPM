import os
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models
# from .clip_fsar import load,tokenize
# from .myRes import Transformer_v1, cos_sim, extract_class_indices

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
            backbone = models.resnet50(pretrained=True)
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
        
        
def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]

class CNN_OTAM_CLIPFSAR(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_OTAM_CLIPFSAR, self).__init__(cfg)
        args = cfg
        self.args = cfg
        if cfg.MODEL.BACKBONE=="RN50":
            backbone, self.preprocess = load(cfg.MODEL.BACKBONE, device="cuda", cfg=cfg, jit=False)   # ViT-B/16
            self.backbone = backbone.visual    # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 1024
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
        
        
        self.mid_layer = nn.Sequential() 
        self.classification_layer = nn.Sequential() 
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)
        
        if hasattr(self.args.MODEL, "TRANSFORMER_DEPTH") and self.args.MODEL.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2, depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
        # set_trace()

    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        if self.training:
            support_features = self.backbone(support_images).squeeze()
            # os.system("nvidia-smi")
            target_features = self.backbone(target_images).squeeze()
            # os.system("nvidia-smi")

            dim = int(support_features.shape[1])

            support_features = support_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)
            target_features = target_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)
            support_features_text = None
            
        else:
            support_features = self.backbone(support_images).squeeze()
            target_features = self.backbone(target_images).squeeze()
            dim = int(target_features.shape[1])
            target_features = target_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)
            support_features = support_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)
            # support_real_class = torch.unique(support_real_class)
            support_features_text = self.text_features_test[support_real_class.long()]


        return support_features, target_features, support_features_text

    def forward(self, inputs):
        support_images, support_labels, target_images, support_real_class = inputs['context_images'], inputs['context_labels'], inputs['target_images'], inputs['real_support_labels'] # [200, 3, 224, 224] inputs["real_support_labels"]
        
        # set_trace()
        if self.training:
            support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
            support_bs = support_features.shape[0]
            target_bs = target_features.shape[0]
            
            
            if hasattr(self.args.MODEL, "USE_CLASSIFICATION") and self.args.MODEL.USE_CLASSIFICATION:
                feature_classification_in = torch.cat([support_features,target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale
            else:
                class_text_logits = None
            
            
            if self.training:
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.SEQ_LEN, 1)
            
            else:
                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.SEQ_LEN, 1) # .repeat(support_bs+target_bs, 1, 1)
            
            target_features = self.context2(target_features, target_features, target_features)
            context_support = self.mid_layer(context_support) 
            if hasattr(self.args.MODEL, "MERGE_BEFORE") and self.args.MODEL.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                context_support = torch.stack(context_support)
            support_features = torch.cat([support_features, context_support], dim=1)
            support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.SEQ_LEN,:]
            if hasattr(self.args.MODEL, "MERGE_BEFORE") and self.args.MODEL.MERGE_BEFORE:
                pass
            else:
                unique_labels = torch.unique(support_labels)
                support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)



            unique_labels = torch.unique(support_labels)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')  
            target_features = rearrange(target_features, 'b s d -> (b s) d')    

            frame_sim = cos_sim(target_features, support_features)  
            frame_dists = 1 - frame_sim
            
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

            # calculate query -> support and support -> query
            if hasattr(self.args.MODEL, "SINGLE_DIRECT") and self.args.MODEL.SINGLE_DIRECT:
                cum_dists = OTAM_cum_dist_v2(dists)
            else:
                cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        
        else:
            if hasattr(self.args.MODEL, "EVAL_TEXT") and self.args.MODEL.EVAL_TEXT:
                unique_labels = torch.unique(support_labels)
                support_features, target_features, text_features = self.get_feats(support_images, target_images, support_real_class) 
                text_features = [torch.mean(torch.index_select(text_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                text_features = torch.stack(text_features)
                # unique_labels = torch.unique(support_labels)
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.scale # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)
                
                cum_dists = -logits_per_image # 
                class_text_logits = None

                
            elif hasattr(self.args.MODEL, "COMBINE") and self.args.MODEL.COMBINE:
                # text_features = self.text_features_test[support_real_class.long()]
                unique_labels = torch.unique(support_labels)
                support_features, target_features, text_features = self.get_feats(support_images, target_images, support_real_class) 
                text_features = [torch.mean(torch.index_select(text_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                text_features = torch.stack(text_features)
                # unique_labels = torch.unique(support_labels)
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.scale # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)
                
                class_text_logits = None

                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]
                
                feature_classification_in = torch.cat([support_features,target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale

                if self.training:
                    context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.SEQ_LEN, 1)
                
                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.SEQ_LEN, 1) # .repeat(support_bs+target_bs, 1, 1)
                
                target_features = self.context2(target_features, target_features, target_features)
                context_support = self.mid_layer(context_support)  # F.relu(self.mid_layer(context_support))
                if hasattr(self.args.MODEL, "MERGE_BEFORE") and self.args.MODEL.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                support_features = torch.cat([support_features, context_support], dim=1)
                support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.SEQ_LEN,:]
                if hasattr(self.args.MODEL, "MERGE_BEFORE") and self.args.MODEL.MERGE_BEFORE:
                    pass
                else:
                    unique_labels = torch.unique(support_labels)
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)



                unique_labels = torch.unique(support_labels)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]

                support_features = rearrange(support_features, 'b s d -> (b s) d')  
                target_features = rearrange(target_features, 'b s d -> (b s) d')    

                frame_sim = cos_sim(target_features, support_features)    
                frame_dists = 1 - frame_sim
                
                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

                # calculate query -> support and support -> query
                if hasattr(self.args.MODEL, "SINGLE_DIRECT") and self.args.MODEL.SINGLE_DIRECT:
                    cum_dists_visual = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists_visual = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
                cum_dists_visual_soft = F.softmax((8-cum_dists_visual)/8., dim=1)
                if hasattr(self.args.MODEL, "TEXT_COFF") and self.args.MODEL.TEXT_COFF:
                    cum_dists = -(logits_per_image.pow(self.args.MODEL.TEXT_COFF)*cum_dists_visual_soft.pow(1.0-self.args.MODEL.TEXT_COFF))
                else:
                    cum_dists = -(logits_per_image.pow(0.9)*cum_dists_visual_soft.pow(0.1))
                
                class_text_logits = None

            else:
                support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]
                
                feature_classification_in = torch.cat([support_features,target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale

                
                if self.training:
                    context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.SEQ_LEN, 1)
                
                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.SEQ_LEN, 1) # .repeat(support_bs+target_bs, 1, 1)
                
                target_features = self.context2(target_features, target_features, target_features)
                if hasattr(self.args.MODEL, "MERGE_BEFORE") and self.args.MODEL.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                support_features = torch.cat([support_features, context_support], dim=1)
                support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.SEQ_LEN,:]
                if hasattr(self.args.MODEL, "MERGE_BEFORE") and self.args.MODEL.MERGE_BEFORE:
                    pass
                else:
                    unique_labels = torch.unique(support_labels)
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)


                unique_labels = torch.unique(support_labels)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]

                support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
                target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]

                frame_sim = cos_sim(target_features, support_features)    # [200, 200]
                frame_dists = 1 - frame_sim
                
                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

                # calculate query -> support and support -> query
                if hasattr(self.args.MODEL, "SINGLE_DIRECT") and self.args.MODEL.SINGLE_DIRECT:
                    cum_dists = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        


        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists.unsqueeze(0), "class_logits": class_text_logits.unsqueeze(0)}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
    

if __name__ == '__main__':
    from clip_fsar import load,tokenize
    from myRes import Transformer_v1, cos_sim, extract_class_indices
    clip_backbone, preprocess = load('RN50', device="cuda", cfg=None, jit=False)  # ViT-B/16
    backbone = clip_backbone.visual.cpu()  # model.load_state_dict(state_dict)
    
    i = torch.rand(8,3,224,224)
    o=backbone(i)
    # print(backbone)
    print(o.shape)
    
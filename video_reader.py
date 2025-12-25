import torch
import sys
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random
import json

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor


class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []
    
    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l

        return max_len

    def __len__(self):
        return len(self.gt_a_list)

"""Dataset for few-shot videos, which returns few-shot tasks. """
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.get_item_counter = 0

        self.train = True
        self.only_test = cfg.TEST.ONLY_TEST

        self.data_dir = cfg.path
        self.annotation_path = cfg.traintestlist
        #self.dataname = cfg.traintestlist.split()

        self.tensor_transform = transforms.ToTensor()

        self.way=cfg.TRAIN.WAY
        self.shot=cfg.TRAIN.SHOT
        self.query_per_class=cfg.TRAIN.QUERY_PER_CLASS
        self.query_per_class_test=cfg.TEST.QUERY_PER_CLASS

        self.norm = None
        self.seq_len = cfg.DATA.SEQ_LEN
        self.img_size = cfg.DATA.IMG_SIZE
        self.img_norm = cfg.DATA.IMG_NORM

        self.train_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self.read_dir()

    def setup_transforms(self):
        video_transform_list = []
        video_test_list = []
            
        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)

        if self.cfg.DATA.DATASET != 'ssv2' and self.cfg.DATA.DATASET != 'ssv2_cmn':
            #print('Add Flip augmentation')
            video_transform_list.append(RandomHorizontalFlip())
        else:
            print('Remove Flip augmentation for SSv2 dataset')

        video_transform_list.append(RandomCrop(self.img_size))
        video_test_list.append(CenterCrop(self.img_size))

        # if self.cfg.DATA.DATASET == 'kinetics':
        #     print('Add data normalization for Kinetics')
        #     self.norm = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)

    """ load the paths of all videos in the train and test splits. """
    def read_dir(self):
        for mode in ["train", "val"]:
            if self.only_test:
                if mode == 'train':
                    continue
                mode = 'test'
            #########################
            else:
                if mode == 'val':
                    mode = 'test'
            #########################
            fname = "{}list{:02d}.txt".format(mode, self.cfg.DATA.SPLIT)
            f = os.path.join(self.annotation_path, fname)
            mode_path = os.path.join(self.data_dir, mode)
            c = self.get_train_or_test_db(mode)
            with open(f, "r") as fid:
                data = fid.readlines()
                cls_dic = {}
                idx = 0
                for line in data:
                    tmp = line.strip().split('/')
                    assert len(tmp) == 2
                    cls, vid_name = tmp
                    vid_path = os.path.join(mode_path, line.strip())
                    imgs_path = []
                    with os.scandir(vid_path) as imgs_e:
                        for img_e in imgs_e:
                            imgs_path.append(img_e.path)
                    if len(imgs_path) < self.seq_len:
                        continue
                    imgs_path.sort()
                    if self.cfg.DATA.DATASET == 'ssv2' or self.cfg.DATA.DATASET == 'ssv2_cmn':
                        class_id = int(cls.split(mode)[-1])
                        if self.cfg.DATA.DATASET == 'ssv2_cmn' and mode == 'train':
                            class_id -= 24
                    else:
                        if cls_dic.get(cls) == None:
                            cls_dic[cls] = idx
                            idx += 1
                        class_id = cls_dic.get(cls)
                    c.add_vid(imgs_path, class_id)
        
        # # 过滤测试集中样本数不足40的类别
        # if self.only_test:
        #     self.filter_test_classes(min_samples=65)
        
        print("loaded {}".format(self.data_dir))
        if self.only_test:
            print("test: {} / {}".format(len(self.test_split), len(self.test_split.get_unique_classes())))
        else:
            print("train: {} / {}, test: {} / {}".format(len(self.train_split), len(self.train_split.get_unique_classes()), len(self.test_split), len(self.test_split.get_unique_classes())))
    
    def filter_test_classes(self, min_samples=60):
        """
        过滤掉测试集中样本数不足指定数量的类别
        
        参数:
            min_samples: 每个类别至少需要的样本数
        """
        # 获取所有测试类别
        all_classes = self.test_split.get_unique_classes()
        # 记录需要保留的视频和标签
        filtered_videos = []
        filtered_labels = []
        
        # 统计每个类别的样本数并过滤
        for cls in all_classes:
            num_samples = self.test_split.get_num_videos_for_class(cls)
            if num_samples >= min_samples:
                # 保留该类别的所有视频
                for i in range(len(self.test_split.gt_a_list)):
                    if self.test_split.gt_a_list[i] == cls:
                        filtered_videos.append(self.test_split.videos[i])
                        filtered_labels.append(cls)
            else:
                print(f"过滤掉测试类别 {cls}，样本数不足: {num_samples} < {min_samples}")
        
        # 更新测试集
        self.test_split.videos = filtered_videos
        self.test_split.gt_a_list = filtered_labels
        
        print(f"过滤后测试集: {len(self.test_split)} 样本，{len(self.test_split.get_unique_classes())} 类别")

    def get_train_or_test_db(self, split=None):
        if split is None:
            get_train_split = self.train
        elif split == 'train':
            get_train_split = True
        elif split == 'test' or split == 'val':
            get_train_split = False
        else:
            print(f'err split: {split}\n')
            sys.exit()
        if get_train_split:
            return self.train_split
        else:
            return self.test_split
            #return self.test_split
   
    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        return len(c)

    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes


    def read_single_image(self, path):
        with Image.open(path) as i:
            i.load()
            return i
    
    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        n_frames = len(paths)
        if n_frames == self.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            if self.train:
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames-1 -excess_pad, n_frames-1)
            else:
                start = 1
                end = n_frames - 2

            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0
            else:
                pass

            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]

            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]
        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]

            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)
        return imgs, vid_id

    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        #select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.query_per_class
        else:
            n_queries = self.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        for bl, bc in enumerate(batch_classes):
            #print(bl, bc)
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.shot + n_queries)

            for idx in idxs[0:self.shot]:
                vid, vid_id = self.get_seq(bc, idx)
                support_set.append(vid)
                support_labels.append(bl)
                real_support_labels.append(bc)
            for idx in idxs[self.shot:]:
                vid, vid_id = self.get_seq(bc, idx)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)

        s = list(zip(support_set, support_labels, real_support_labels))
        random.shuffle(s)
        support_set, support_labels, real_support_labels = zip(*s)

        t = list(zip(target_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)

        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_support_labels = torch.FloatTensor(real_support_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes)
        #print(support_labels)
        return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, \
                "target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes, "real_support_labels":real_support_labels}
